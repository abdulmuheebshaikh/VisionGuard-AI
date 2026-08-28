from flask import Flask, render_template, request, jsonify
from pathlib import Path
from PIL import Image, ImageStat, ImageFilter
import os
import uuid
import datetime
import traceback
import numpy as np
import tensorflow as tf


# ============================================================
# VISIONGUARD AI - RENDER-OPTIMIZED FLASK BACKEND
# ============================================================

app = Flask(__name__)

# ------------------------------------------------------------
# RENDER / CPU CONFIGURATION
# ------------------------------------------------------------

# Limit TensorFlow CPU threads so the free Render instance
# does not get overloaded.
try:
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass


# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "visionguard_model.keras"

UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
GRADCAM_FOLDER = BASE_DIR / "static" / "gradcam"

ALLOWED_EXTENSIONS = {
    "png",
    "jpg",
    "jpeg",
    "webp"
}

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
GRADCAM_FOLDER.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# MODEL CLASSES
# ------------------------------------------------------------
# IMPORTANT:
# Keep this order exactly the same as the model training order.

CLASS_NAMES = [
    "Mild",
    "Moderate",
    "No_DR",
    "Proliferate_DR",
    "Severe"
]


# ------------------------------------------------------------
# GLOBAL MODEL VARIABLES
# ------------------------------------------------------------

model = None
gradcam_model = None
last_conv_layer = None


# ------------------------------------------------------------
# LOAD CNN MODEL
# ------------------------------------------------------------

print("=" * 70)
print("VISIONGUARD AI")
print("Loading CNN model...")
print("Model path:", MODEL_PATH)
print("=" * 70)

try:

    model = tf.keras.models.load_model(
        str(MODEL_PATH),
        compile=False
    )

    print("CNN MODEL LOADED SUCCESSFULLY")
    print("Input shape:", model.input_shape)
    print("Output shape:", model.output_shape)

except Exception as error:

    model = None

    print("=" * 70)
    print("ERROR: CNN MODEL COULD NOT BE LOADED")
    print(error)
    traceback.print_exc()
    print("=" * 70)


# ------------------------------------------------------------
# FIND LAST CONVOLUTION LAYER
# ------------------------------------------------------------

def find_last_conv_layer(keras_model):

    candidates = []

    for layer in keras_model.layers:

        try:

            shape = layer.output.shape

            if len(shape) != 4:
                continue

            layer_type = layer.__class__.__name__.lower()

            if (
                "conv" in layer_type
                or "separable" in layer_type
                or "depthwise" in layer_type
            ):

                candidates.append(layer)

        except Exception:
            continue

    if candidates:
        return candidates[-1]

    # Fallback:
    # use the last 4-D feature-map layer.

    for layer in reversed(keras_model.layers):

        try:

            if len(layer.output.shape) == 4:
                return layer

        except Exception:
            continue

    return None


# ------------------------------------------------------------
# PREPARE GRAD-CAM MODEL ONCE
# ------------------------------------------------------------

def prepare_gradcam_model():

    global gradcam_model
    global last_conv_layer

    if model is None:
        return

    try:

        last_conv_layer = find_last_conv_layer(model)

        if last_conv_layer is None:

            print(
                "WARNING: No convolution layer found."
            )

            gradcam_model = None
            return

        print(
            "Grad-CAM layer:",
            last_conv_layer.name
        )

        gradcam_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[
                last_conv_layer.output,
                model.output
            ]
        )

        print(
            "Grad-CAM model prepared successfully."
        )

    except Exception as error:

        gradcam_model = None

        print(
            "WARNING: Could not prepare Grad-CAM model:"
        )

        print(error)


prepare_gradcam_model()


# ------------------------------------------------------------
# BASIC HELPERS
# ------------------------------------------------------------

def allowed_file(filename):

    if not filename:
        return False

    if "." not in filename:
        return False

    extension = filename.rsplit(
        ".",
        1
    )[1].lower()

    return extension in ALLOWED_EXTENSIONS


def make_screening_id():

    date_part = datetime.datetime.now().strftime(
        "%Y%m%d"
    )

    random_part = uuid.uuid4().hex[:6].upper()

    return f"SCR-{date_part}-{random_part}"


def get_input_size():

    try:

        shape = model.input_shape

        if isinstance(shape, list):
            shape = shape[0]

        height = shape[1]
        width = shape[2]

        if height and width:

            return (
                int(width),
                int(height)
            )

    except Exception:

        pass

    return 224, 224


def prepare_image(img):

    width, height = get_input_size()

    resized = img.convert(
        "RGB"
    ).resize(
        (width, height),
        Image.Resampling.LANCZOS
    )

    array = np.asarray(
        resized
    ).astype(
        "float32"
    ) / 255.0

    array = np.expand_dims(
        array,
        axis=0
    )

    return array


# ------------------------------------------------------------
# IMAGE QUALITY CHECK
# ------------------------------------------------------------

def quality_check(img):

    x = img.convert(
        "RGB"
    ).resize(
        (256, 256)
    )

    stats = ImageStat.Stat(x)

    brightness = sum(
        stats.mean
    ) / 3.0

    edges = x.filter(
        ImageFilter.FIND_EDGES
    )

    edge_stats = ImageStat.Stat(
        edges
    )

    sharpness = sum(
        edge_stats.var
    ) / 3.0

    if brightness < 35:

        return {
            "score": 28,
            "status": "poor",
            "message":
                "Too dark. Capture a brighter, centered fundus image."
        }

    if brightness > 225:

        return {
            "score": 35,
            "status": "poor",
            "message":
                "Overexposed. Reduce glare and recapture."
        }

    if sharpness < 18:

        return {
            "score": 42,
            "status": "poor",
            "message":
                "Image appears blurry. Hold the camera steady and recapture."
        }

    score = int(
        70 + sharpness / 8
    )

    score = min(
        98,
        max(70, score)
    )

    return {
        "score": score,
        "status": "good",
        "message":
            "Image quality is acceptable for AI screening."
    }


# ------------------------------------------------------------
# RISK / EXPLANATION
# ------------------------------------------------------------

def get_risk_message(predicted_class):

    if predicted_class == "No_DR":

        return {
            "risk": "Low screening indication",
            "explanation":
                "The trained AI model classified this image as "
                "No diabetic retinopathy."
        }

    if predicted_class == "Mild":

        return {
            "risk": "Possible mild changes",
            "explanation":
                "The trained AI model classified this image as "
                "Mild diabetic retinopathy."
        }

    if predicted_class == "Moderate":

        return {
            "risk": "Possible moderate changes",
            "explanation":
                "The trained AI model classified this image as "
                "Moderate diabetic retinopathy."
        }

    if predicted_class == "Severe":

        return {
            "risk": "Possible severe changes",
            "explanation":
                "The trained AI model classified this image as "
                "Severe diabetic retinopathy."
        }

    if predicted_class == "Proliferate_DR":

        return {
            "risk": "Possible advanced changes",
            "explanation":
                "The trained AI model classified this image as "
                "Proliferative diabetic retinopathy."
        }

    return {
        "risk":
            "Further ophthalmic evaluation recommended",
        "explanation":
            "The trained AI model produced a retinal screening result."
    }


def get_finding(predicted_class):

    if predicted_class == "No_DR":
        return (
            "No obvious diabetic-retinopathy finding "
            "predicted by the model."
        )

    if predicted_class == "Mild":
        return (
            "Possible mild retinal changes detected."
        )

    if predicted_class == "Moderate":
        return (
            "Possible moderate retinal changes detected."
        )

    if predicted_class == "Severe":
        return (
            "Possible severe retinal changes detected."
        )

    if predicted_class == "Proliferate_DR":
        return (
            "Possible advanced retinal changes detected."
        )

    return (
        "Possible retinal changes detected."
    )


# ------------------------------------------------------------
# GRAD-CAM
# ------------------------------------------------------------

def create_gradcam_heatmap(
    input_tensor,
    target_class_index
):

    if model is None:

        raise RuntimeError(
            "CNN model is not loaded."
        )

    if gradcam_model is None:

        raise RuntimeError(
            "Grad-CAM model is unavailable."
        )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = (
            gradcam_model(
                input_tensor,
                training=False
            )
        )

        if isinstance(
            predictions,
            (list, tuple)
        ):

            predictions = predictions[0]

        class_score = predictions[
            :,
            target_class_index
        ]

    gradients = tape.gradient(
        class_score,
        conv_outputs
    )

    if gradients is None:

        raise RuntimeError(
            "Grad-CAM gradients could not be calculated."
        )

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

    maximum = tf.reduce_max(
        heatmap
    )

    heatmap = tf.where(
        maximum > 0,
        heatmap / maximum,
        heatmap
    )

    return heatmap.numpy()


def save_gradcam_overlay(
    original_img,
    heatmap,
    output_path
):

    original = original_img.convert(
        "RGB"
    )

    heatmap_image = Image.fromarray(
        np.uint8(
            np.clip(
                heatmap,
                0,
                1
            ) * 255
        )
    )

    heatmap_image = heatmap_image.resize(
        original.size,
        Image.Resampling.BILINEAR
    )

    h = (
        np.asarray(
            heatmap_image
        ).astype(
            "float32"
        ) / 255.0
    )

    red = np.clip(
        2.0 * h,
        0,
        1
    )

    green = np.clip(
        2.0 * h - 0.5,
        0,
        1
    )

    blue = np.clip(
        2.0 * h - 1.0,
        0,
        1
    )

    color_map = np.stack(
        [
            red,
            green,
            blue
        ],
        axis=-1
    )

    color_map = Image.fromarray(
        np.uint8(
            color_map * 255
        )
    )

    overlay = Image.blend(
        original,
        color_map,
        alpha=0.42
    )

    overlay.save(
        output_path,
        "JPEG",
        quality=90
    )


# ------------------------------------------------------------
# HOME PAGE
# ------------------------------------------------------------

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ------------------------------------------------------------
# ANALYZE IMAGE
# ------------------------------------------------------------

@app.route(
    "/analyze",
    methods=["POST"]
)
def analyze():

    start_time = datetime.datetime.now()

    try:

        print("")
        print("=" * 70)
        print("NEW VISIONGUARD ANALYSIS REQUEST")
        print("=" * 70)

        # ----------------------------------------------------
        # MODEL CHECK
        # ----------------------------------------------------

        if model is None:

            return jsonify({
                "error":
                    "CNN model is not loaded. "
                    "Check visionguard_model.keras."
            }), 500

        # ----------------------------------------------------
        # GET IMAGE
        # ----------------------------------------------------

        file = request.files.get(
            "image"
        )

        if not file:

            return jsonify({
                "error":
                    "No retinal image uploaded."
            }), 400

        if not allowed_file(
            file.filename
        ):

            return jsonify({
                "error":
                    "Invalid image type. "
                    "Use JPG, JPEG, PNG or WEBP."
            }), 400

        extension = (
            file.filename
            .rsplit(".", 1)[1]
            .lower()
        )

        # ----------------------------------------------------
        # OPEN IMAGE
        # ----------------------------------------------------

        try:

            img = Image.open(
                file.stream
            ).convert("RGB")

        except Exception:

            return jsonify({
                "error":
                    "Invalid or corrupted image."
            }), 400

        print(
            "Image received:",
            file.filename
        )

        print(
            "Image size:",
            img.size
        )

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

        if not patient_id:

            patient_id = (
                "P-" +
                uuid.uuid4().hex[:6].upper()
            )

        screening_id = make_screening_id()

        # ----------------------------------------------------
        # QUALITY CHECK
        # ----------------------------------------------------

        print(
            "Running image quality check..."
        )

        quality = quality_check(
            img
        )

        print(
            "Quality:",
            quality
        )

        # ----------------------------------------------------
        # SAVE ORIGINAL IMAGE
        # ----------------------------------------------------

        unique_name = (
            f"{screening_id}_"
            f"{uuid.uuid4().hex[:8]}."
            f"{extension}"
        )

        original_path = (
            UPLOAD_FOLDER /
            unique_name
        )

        if extension in {
            "jpg",
            "jpeg"
        }:

            img.save(
                original_path,
                "JPEG",
                quality=90
            )

        elif extension == "webp":

            img.save(
                original_path,
                "WEBP",
                quality=90
            )

        else:

            img.save(
                original_path
            )

        image_url = (
            "/static/uploads/"
            + unique_name
        )

        # ----------------------------------------------------
        # PREPARE IMAGE
        # ----------------------------------------------------

        print(
            "Preparing image for CNN..."
        )

        ai_img = prepare_image(
            img
        )

        print(
            "Prepared tensor:",
            ai_img.shape
        )

        # ----------------------------------------------------
        # CNN PREDICTION
        # ----------------------------------------------------

        print(
            "Starting CNN prediction..."
        )

        prediction_start = (
            datetime.datetime.now()
        )

        predictions = model.predict(
            ai_img,
            verbose=0
        )

        prediction_time = (
            datetime.datetime.now()
            - prediction_start
        ).total_seconds()

        print(
            "CNN prediction completed in",
            round(prediction_time, 2),
            "seconds"
        )

        # ----------------------------------------------------
        # NORMALIZE OUTPUT
        # ----------------------------------------------------

        if isinstance(
            predictions,
            (list, tuple)
        ):

            predictions = predictions[0]

        predictions = np.asarray(
            predictions
        )

        if predictions.ndim == 1:

            predictions = np.expand_dims(
                predictions,
                axis=0
            )

        # ----------------------------------------------------
        # CHECK MODEL OUTPUT
        # ----------------------------------------------------

        if (
            predictions.shape[-1]
            != len(CLASS_NAMES)
        ):

            return jsonify({
                "error":
                    "Model output classes do not match "
                    "CLASS_NAMES. "
                    f"Model returned "
                    f"{predictions.shape[-1]} outputs, "
                    f"but the application expects "
                    f"{len(CLASS_NAMES)}."
            }), 500

        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        predicted_index = int(
            np.argmax(
                predictions[0]
            )
        )

        predicted_class = (
            CLASS_NAMES[
                predicted_index
            ]
        )

        confidence_value = float(
            predictions[0][
                predicted_index
            ]
        )

        # If the model returns logits instead of probabilities,
        # convert them to probabilities.

        if (
            confidence_value < 0
            or confidence_value > 1
            or not np.isclose(
                np.sum(
                    predictions[0]
                ),
                1.0,
                atol=0.05
            )
        ):

            probability_values = tf.nn.softmax(
                tf.convert_to_tensor(
                    predictions[0],
                    dtype=tf.float32
                )
            ).numpy()

            confidence_value = float(
                probability_values[
                    predicted_index
                ]
            )

        confidence = round(
            confidence_value * 100,
            2
        )

        print(
            "Predicted class:",
            predicted_class
        )

        print(
            "Confidence:",
            confidence,
            "%"
        )

        # ----------------------------------------------------
        # RISK / EXPLANATION
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

        # Temporarily disabled on Render to reduce CPU/memory usage.
        # CNN prediction will still be performed normally.

        gradcam_url = None

        gradcam_status = "disabled"

        gradcam_message = (
            "Grad-CAM temporarily disabled to reduce server resource usage."
        )

        print("Grad-CAM skipped.")

        # ----------------------------------------------------
        # TIMESTAMP
        # ----------------------------------------------------

        timestamp = (
            datetime.datetime.now()
            .strftime(
                "%d %b %Y, %I:%M %p"
            )
        )

        # ----------------------------------------------------
        # TOTAL TIME
        # ----------------------------------------------------

        total_time = (
            datetime.datetime.now()
            - start_time
        ).total_seconds()

        print(
            "Total analysis time:",
            round(total_time, 2),
            "seconds"
        )

        # ----------------------------------------------------
        # RESPONSE
        # ----------------------------------------------------

        response = {

            "mode":
                "AI MODEL",

            "screeningId":
                screening_id,

            "patient": {

                "id":
                    patient_id,

                "name":
                    patient_name,

                "age":
                    patient_age,

                "gender":
                    patient_gender
            },

            "quality":
                quality,

            "image_url":
                image_url,

            "gradcam_url":
                gradcam_url,

            "gradcam": {

                "status":
                    gradcam_status,

                "message":
                    gradcam_message
            },

            "class_":
                predicted_class,

            "confidence":
                confidence,

            "risk":
                risk_info["risk"],

            "explanation":
                risk_info["explanation"],

            "lesions":
                lesions,

            "timestamp":
                timestamp
        }

        # ----------------------------------------------------
        # FINAL LOG
        # ----------------------------------------------------

        print("=" * 70)
        print("SCREENING COMPLETED SUCCESSFULLY")
        print("Screening ID:", screening_id)
        print("Patient ID:", patient_id)
        print("Result:", predicted_class)
        print("Confidence:", confidence, "%")
        print("Quality:", quality["score"])
        print("Grad-CAM:", gradcam_status)
        print("Total time:", round(total_time, 2), "seconds")
        print("=" * 70)

        return jsonify(
            response
        )

    except Exception as error:

        print("=" * 70)
        print("UNEXPECTED ERROR IN /analyze")
        print(error)
        traceback.print_exc()
        print("=" * 70)

        return jsonify({

            "error":
                "An error occurred while analyzing "
                "the image.",

            "details":
                str(error)

        }), 500


# ------------------------------------------------------------
# HEALTH CHECK
# ------------------------------------------------------------

@app.route("/health")
def health():

    return jsonify({

        "status":
            "online"
            if model is not None
            else "model_error",

        "service":
            "VisionGuard AI",

        "model":
            "loaded"
            if model is not None
            else "not loaded",

        "classes":
            CLASS_NAMES,

        "gradcam":"disabled"
            if gradcam_model is not None
            else "unavailable"
    })


# ------------------------------------------------------------
# FILE TOO LARGE
# ------------------------------------------------------------

@app.errorhandler(413)
def file_too_large(error):

    return jsonify({

        "error":
            "Image is too large. "
            "Maximum size is 10 MB."

    }), 413


# ------------------------------------------------------------
# LOCAL RUN
# ------------------------------------------------------------

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            "5000"
        )
    )

    app.run(
        debug=False,
        host="0.0.0.0",
        port=port
    )
