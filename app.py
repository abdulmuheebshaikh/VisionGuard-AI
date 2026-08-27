from flask import Flask, render_template, request, jsonify
from PIL import Image, ImageStat, ImageFilter
import os
import uuid
import datetime
import traceback
import numpy as np
import tensorflow as tf

# ============================================================
# VISIONGUARD AI - FLASK BACKEND
# Features:
#   - CNN/Keras diabetic-retinopathy classification
#   - Image quality gate
#   - Grad-CAM explainability
#   - Patient information
#   - Patient ID + separate Screening ID
#   - Saved original image + Grad-CAM image
#   - JSON response compatible with the existing app.js
#
# IMPORTANT:
#   Keep CLASS_NAMES in EXACTLY the same order used while
#   training visionguard_model.keras.
# ============================================================

app = Flask(__name__)

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

MODEL_PATH = "visionguard_model.keras"

UPLOAD_FOLDER = os.path.join("static", "uploads")
GRADCAM_FOLDER = os.path.join("static", "gradcam")

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GRADCAM_FOLDER, exist_ok=True)


# ------------------------------------------------------------
# MODEL CLASSES
# ------------------------------------------------------------

CLASS_NAMES = [
    "Mild",
    "Moderate",
    "No_DR",
    "Proliferate_DR",
    "Severe"
]


# ------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------

print("=" * 60)
print("VisionGuard AI - loading CNN model...")
print("Model:", MODEL_PATH)

try:
    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )
    print("VisionGuard AI CNN model loaded successfully.")
    print("Model input:", model.input_shape)
    print("Model output:", model.output_shape)
except Exception as error:
    model = None
    print("ERROR: Could not load the model.")
    print(error)
    print("Make sure visionguard_model.keras is in the same folder as app.py.")


# ------------------------------------------------------------
# BASIC HELPERS
# ------------------------------------------------------------

def allowed_file(filename):
    if not filename or "." not in filename:
        return False

    extension = filename.rsplit(".", 1)[1].lower()
    return extension in ALLOWED_EXTENSIONS


def format_class_name(class_name):
    return str(class_name).replace("_", " ")


def make_screening_id():
    """
    Creates a separate screening ID for every analysis.

    Example:
        SCR-20260827-A1B2C3
    """
    date_part = datetime.datetime.now().strftime("%Y%m%d")
    random_part = uuid.uuid4().hex[:6].upper()
    return f"SCR-{date_part}-{random_part}"


def get_input_size():
    """
    Reads the CNN input size from the loaded model.
    Falls back to 224x224 if the model uses a dynamic/unknown size.
    """
    try:
        shape = model.input_shape

        if isinstance(shape, list):
            shape = shape[0]

        height = shape[1]
        width = shape[2]

        if height and width:
            return int(width), int(height)
    except Exception:
        pass

    return 224, 224


def prepare_image(img):
    """
    Same basic preprocessing used by the current application:
    RGB -> model input size -> float32 -> [0,1] -> batch dimension.
    """
    width, height = get_input_size()

    resized = img.convert("RGB").resize(
        (width, height),
        Image.Resampling.LANCZOS
    )

    array = np.asarray(resized).astype("float32") / 255.0
    array = np.expand_dims(array, axis=0)

    return array


# ------------------------------------------------------------
# IMAGE QUALITY GATE
# ------------------------------------------------------------

def quality_check(img):
    """
    Basic quality gate.

    Checks:
      - brightness
      - sharpness

    This is a screening-quality check, not a medical diagnosis.
    """

    x = img.convert("RGB").resize((256, 256))

    stats = ImageStat.Stat(x)
    brightness = sum(stats.mean) / 3.0

    edges = x.filter(ImageFilter.FIND_EDGES)
    edge_stats = ImageStat.Stat(edges)
    sharpness = sum(edge_stats.var) / 3.0

    if brightness < 35:
        return {
            "score": 28,
            "status": "poor",
            "message": (
                "Too dark. Capture a brighter, centered fundus image."
            )
        }

    if brightness > 225:
        return {
            "score": 35,
            "status": "poor",
            "message": (
                "Overexposed. Reduce glare and recapture."
            )
        }

    if sharpness < 18:
        return {
            "score": 42,
            "status": "poor",
            "message": (
                "Image appears blurry. Hold the camera steady and recapture."
            )
        }

    score = int(70 + sharpness / 8)
    score = min(98, max(70, score))

    return {
        "score": score,
        "status": "good",
        "message": (
            "Image quality is acceptable for AI screening."
        )
    }


# ------------------------------------------------------------
# RISK / EXPLANATION
# ------------------------------------------------------------

def get_risk_message(predicted_class):
    class_name = format_class_name(predicted_class).lower()

    if class_name == "no dr":
        return {
            "risk": "Low screening indication",
            "explanation": (
                "The trained AI model classified this image as "
                "No diabetic retinopathy."
            )
        }

    if class_name == "mild":
        return {
            "risk": "Possible mild changes",
            "explanation": (
                "The trained AI model classified this image as "
                "Mild diabetic retinopathy."
            )
        }

    if class_name == "moderate":
        return {
            "risk": "Possible moderate changes",
            "explanation": (
                "The trained AI model classified this image as "
                "Moderate diabetic retinopathy."
            )
        }

    if class_name == "severe":
        return {
            "risk": "Possible severe changes",
            "explanation": (
                "The trained AI model classified this image as "
                "Severe diabetic retinopathy."
            )
        }

    if class_name == "proliferate dr":
        return {
            "risk": "Possible advanced changes",
            "explanation": (
                "The trained AI model classified this image as "
                "Proliferative diabetic retinopathy."
            )
        }

    return {
        "risk": "Further ophthalmic evaluation recommended",
        "explanation": (
            "The trained AI model produced a retinal screening result."
        )
    }


def get_finding(predicted_class):
    if predicted_class == "No_DR":
        return "No obvious diabetic-retinopathy finding predicted by the model."

    if predicted_class == "Mild":
        return "Possible mild retinal changes detected."

    if predicted_class == "Moderate":
        return "Possible moderate retinal changes detected."

    if predicted_class == "Severe":
        return "Possible severe retinal changes detected."

    if predicted_class == "Proliferate_DR":
        return "Possible advanced retinal changes detected."

    return "Possible retinal changes detected."


# ------------------------------------------------------------
# GRAD-CAM
# ------------------------------------------------------------

def find_last_conv_layer(keras_model):
    """
    Automatically finds the last 2D convolution-like layer.

    This avoids hard-coding a layer name such as 'conv5_block16_concat',
    because different CNN architectures use different names.
    """

    candidates = []

    for layer in keras_model.layers:
        try:
            output_shape = layer.output.shape

            if len(output_shape) == 4:
                layer_name = layer.__class__.__name__.lower()

                if (
                    "conv" in layer_name
                    or "separable" in layer_name
                    or "depthwise" in layer_name
                ):
                    candidates.append(layer)
        except Exception:
            continue

    if not candidates:
        # Second fallback: any 4-D feature-map layer.
        for layer in reversed(keras_model.layers):
            try:
                if len(layer.output.shape) == 4:
                    candidates.append(layer)
                    break
            except Exception:
                continue

    if not candidates:
        return None

    return candidates[-1]


def gradcam_heatmap(input_tensor, target_class_index=None):
    """
    Creates a Grad-CAM heatmap for the predicted class.

    Returns:
        numpy array in the range [0, 1]
    """

    if model is None:
        raise RuntimeError("CNN model is not loaded.")

    last_conv_layer = find_last_conv_layer(model)

    if last_conv_layer is None:
        raise RuntimeError(
            "Could not find a convolutional feature layer for Grad-CAM."
        )

    # Functional models can expose the layer directly.
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            last_conv_layer.output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(input_tensor)

        # Some Keras models return a list/tuple of outputs.
        if isinstance(predictions, (list, tuple)):
            predictions = predictions[0]

        if target_class_index is None:
            target_class_index = tf.argmax(
                predictions[0]
            )

        class_score = predictions[:, target_class_index]

    gradients = tape.gradient(
        class_score,
        conv_outputs
    )

    if gradients is None:
        raise RuntimeError(
            "Grad-CAM gradients could not be calculated."
        )

    # Global average pooling of gradients.
    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(1, 2)
    )

    conv_outputs = conv_outputs[0]
    pooled_gradients = pooled_gradients[0]

    heatmap = tf.reduce_sum(
        conv_outputs * pooled_gradients,
        axis=-1
    )

    heatmap = tf.maximum(
        heatmap,
        0
    )

    maximum = tf.reduce_max(heatmap)

    heatmap = tf.where(
        maximum > 0,
        heatmap / maximum,
        heatmap
    )

    return heatmap.numpy()


def save_gradcam_overlay(original_img, heatmap, output_path):
    """
    Saves:
      original fundus image + Grad-CAM heatmap overlay

    The saved image can be displayed directly by the frontend.
    """

    original = original_img.convert("RGB")

    heatmap_image = Image.fromarray(
        np.uint8(np.clip(heatmap, 0, 1) * 255)
    )

    heatmap_image = heatmap_image.resize(
        original.size,
        Image.Resampling.BILINEAR
    )

    # Red/yellow-style attention map without requiring OpenCV.
    h = np.asarray(heatmap_image).astype("float32") / 255.0

    # Create a simple RGB heatmap.
    red = np.clip(2.0 * h, 0, 1)
    green = np.clip(2.0 * h - 0.5, 0, 1)
    blue = np.clip(2.0 * h - 1.0, 0, 1)

    color_map = np.stack(
        [red, green, blue],
        axis=-1
    )

    color_map = Image.fromarray(
        np.uint8(color_map * 255)
    )

    # Blend attention map over the original fundus.
    overlay = Image.blend(
        original,
        color_map,
        alpha=0.42
    )

    overlay.save(
        output_path,
        "JPEG",
        quality=92
    )


# ------------------------------------------------------------
# HOME
# ------------------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")


# ------------------------------------------------------------
# AI ANALYSIS
# ------------------------------------------------------------

@app.route("/analyze", methods=["POST"])
def analyze():

    try:
        # ----------------------------------------------------
        # MODEL CHECK
        # ----------------------------------------------------

        if model is None:
            return jsonify({
                "error": (
                    "CNN model is not loaded. "
                    "Check visionguard_model.keras."
                )
            }), 500

        # ----------------------------------------------------
        # IMAGE
        # ----------------------------------------------------

        file = request.files.get("image")

        if not file:
            return jsonify({
                "error": "No retinal image uploaded."
            }), 400

        if not allowed_file(file.filename):
            return jsonify({
                "error": (
                    "Invalid image type. Use JPG, JPEG, PNG or WEBP."
                )
            }), 400

        extension = file.filename.rsplit(
            ".", 1
        )[1].lower()

        try:
            img = Image.open(
                file.stream
            ).convert("RGB")
        except Exception:
            return jsonify({
                "error": "Invalid or corrupted image."
            }), 400

        # ----------------------------------------------------
        # PATIENT INFORMATION
        # ----------------------------------------------------

        patient_name = request.form.get(
            "patientName",
            ""
        ).strip()

        patient_age = request.form.get(
            "age",
            ""
        ).strip()

        patient_gender = request.form.get(
            "gender",
            ""
        ).strip()

        patient_id = request.form.get(
            "patientId",
            ""
        ).strip()

        # Generate a patient ID if frontend did not send one.
        if not patient_id:
            patient_id = "P-" + uuid.uuid4().hex[:6].upper()

        # Separate screening ID.
        screening_id = make_screening_id()

        # ----------------------------------------------------
        # QUALITY CHECK
        # ----------------------------------------------------

        quality = quality_check(img)

        # ----------------------------------------------------
        # SAVE ORIGINAL IMAGE
        # ----------------------------------------------------

        unique_name = (
            f"{screening_id}_{uuid.uuid4().hex[:8]}.{extension}"
        )

        original_path = os.path.join(
            UPLOAD_FOLDER,
            unique_name
        )

        if extension in {"jpg", "jpeg"}:
            img.save(
                original_path,
                "JPEG",
                quality=92
            )
        elif extension == "webp":
            img.save(
                original_path,
                "WEBP",
                quality=92
            )
        else:
            img.save(original_path)

        image_url = f"/static/uploads/{unique_name}"

        # ----------------------------------------------------
        # PREPARE IMAGE
        # ----------------------------------------------------

        ai_img = prepare_image(img)

        # ----------------------------------------------------
        # CNN PREDICTION
        # ----------------------------------------------------

        predictions = model.predict(
            ai_img,
            verbose=0
        )

        if isinstance(predictions, (list, tuple)):
            predictions = predictions[0]

        predictions = np.asarray(predictions)

        if predictions.ndim == 1:
            predictions = np.expand_dims(
                predictions,
                axis=0
            )

        # ----------------------------------------------------
        # SAFETY CHECK
        # ----------------------------------------------------

        if predictions.shape[-1] != len(CLASS_NAMES):
            return jsonify({
                "error": (
                    "Model output classes do not match CLASS_NAMES. "
                    f"Model returned {predictions.shape[-1]} outputs, "
                    f"but CLASS_NAMES contains {len(CLASS_NAMES)} classes."
                )
            }), 500

        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        predicted_index = int(
            np.argmax(predictions[0])
        )

        predicted_class = CLASS_NAMES[
            predicted_index
        ]

        confidence = float(
            predictions[0][predicted_index] * 100
        )

        confidence = round(
            confidence,
            2
        )

        # ----------------------------------------------------
        # RISK + EXPLANATION
        # ----------------------------------------------------

        risk_info = get_risk_message(
            predicted_class
        )

        finding = get_finding(
            predicted_class
        )

        lesions = [finding]

        # ----------------------------------------------------
        # GRAD-CAM
        # ----------------------------------------------------

        gradcam_url = None
        gradcam_status = "unavailable"
        gradcam_message = ""

        try:
            heatmap = gradcam_heatmap(
                tf.convert_to_tensor(ai_img),
                target_class_index=predicted_index
            )

            gradcam_filename = (
                f"{screening_id}_gradcam.jpg"
            )

            gradcam_path = os.path.join(
                GRADCAM_FOLDER,
                gradcam_filename
            )

            save_gradcam_overlay(
                img,
                heatmap,
                gradcam_path
            )

            gradcam_url = f"/static/gradcam/{gradcam_filename}"

            gradcam_status = "generated"
            gradcam_message = (
                "Grad-CAM attention map generated for the predicted class."
            )

            print(
                "Grad-CAM generated using:",
                find_last_conv_layer(model).name
            )

        except Exception as gradcam_error:
            # Do not fail the entire screening just because Grad-CAM
            # cannot be generated for an unusual model architecture.
            gradcam_status = "unavailable"
            gradcam_message = str(gradcam_error)

            print(
                "Grad-CAM warning:",
                gradcam_error
            )

        # ----------------------------------------------------
        # TIMESTAMP
        # ----------------------------------------------------

        timestamp = datetime.datetime.now().strftime(
            "%d %b %Y, %I:%M %p"
        )

        # ----------------------------------------------------
        # RESPONSE
        # ----------------------------------------------------

        response = {
            "mode": "AI MODEL",

            # Separate IDs
            "screeningId": screening_id,

            "patient": {
                "id": patient_id,
                "name": patient_name,
                "age": patient_age,
                "gender": patient_gender
            },

            # Image quality
            "quality": quality,

            # Original image
            "image_url": image_url,

            # Grad-CAM image
            "gradcam_url": gradcam_url,

            "gradcam": {
                "status": gradcam_status,
                "message": gradcam_message
            },

            # AI result
            "class_": predicted_class,
            "confidence": confidence,

            "risk": risk_info["risk"],
            "explanation": risk_info["explanation"],

            "lesions": lesions,

            "timestamp": timestamp
        }

        print("=" * 60)
        print("SCREENING COMPLETED")
        print("Screening ID:", screening_id)
        print("Patient ID:", patient_id)
        print("Patient:", patient_name)
        print("Age:", patient_age)
        print("Gender:", patient_gender)
        print("Result:", predicted_class)
        print("Confidence:", confidence)
        print("Quality:", quality["score"])
        print("Original:", image_url)
        print("Grad-CAM:", gradcam_url)
        print("=" * 60)

        return jsonify(response)

    except Exception as error:

        print("=" * 60)
        print("UNEXPECTED ERROR IN /analyze")
        print(error)
        traceback.print_exc()
        print("=" * 60)

        return jsonify({
            "error": (
                "An error occurred while analyzing the image. "
                "Check the Flask terminal for details."
            ),
            "details": str(error)
        }), 500


# ------------------------------------------------------------
# HEALTH CHECK
# ------------------------------------------------------------

@app.route("/health")
def health():

    model_loaded = model is not None

    return jsonify({
        "status": "online" if model_loaded else "model_error",
        "service": "VisionGuard AI",
        "model": "loaded" if model_loaded else "not loaded",
        "classes": CLASS_NAMES,
        "gradcam": "enabled"
    })


# ------------------------------------------------------------
# ERROR: FILE TOO LARGE
# ------------------------------------------------------------

@app.errorhandler(413)
def file_too_large(error):
    return jsonify({
        "error": "Image is too large. Maximum size is 10 MB."
    }), 413


# ------------------------------------------------------------
# RUN
# ------------------------------------------------------------

if __name__ == "__main__":
    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )
