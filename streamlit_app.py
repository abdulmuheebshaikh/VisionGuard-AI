import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import io
import base64
from datetime import datetime, date
import pandas as pd

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="VisionGuard AI",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# FILE PATHS
# ============================================================

MODEL_PATH = "visionguard_model.keras"
LOGO_PATH = os.path.join("static", "visionguard-logo.png")

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

.stApp {
    background: #f6f9fc;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

/* Sidebar */

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #071b33 0%, #0b2945 100%);
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

.sidebar-title {
    font-size: 24px;
    font-weight: 700;
    margin-top: 5px;
}

.sidebar-subtitle {
    font-size: 12px;
    opacity: 0.75;
    margin-bottom: 25px;
}

/* Header */

.main-title {
    font-size: 36px;
    font-weight: 800;
    color: #102a43;
    margin-bottom: 4px;
}

.main-subtitle {
    color: #627d98;
    font-size: 15px;
    margin-bottom: 25px;
}

/* Cards */

.card {
    background: white;
    border-radius: 18px;
    padding: 24px;
    box-shadow: 0 5px 20px rgba(16,42,67,0.07);
    border: 1px solid #e8eef4;
    margin-bottom: 20px;
}

.card-title {
    font-size: 20px;
    font-weight: 700;
    color: #102a43;
    margin-bottom: 8px;
}

.card-subtitle {
    font-size: 13px;
    color: #829ab1;
    margin-bottom: 18px;
}

/* Result */

.result-normal {
    padding: 20px;
    border-radius: 14px;
    background: #e9f8f1;
    border: 1px solid #b8e8d0;
}

.result-warning {
    padding: 20px;
    border-radius: 14px;
    background: #fff6df;
    border: 1px solid #f1d58a;
}

.result-danger {
    padding: 20px;
    border-radius: 14px;
    background: #fff0f0;
    border: 1px solid #efb0b0;
}

.result-title {
    font-size: 23px;
    font-weight: 800;
}

.metric-card {
    background: white;
    border: 1px solid #e5edf4;
    border-radius: 14px;
    padding: 18px;
    text-align: center;
}

.metric-value {
    font-size: 26px;
    font-weight: 800;
    color: #102a43;
}

.metric-label {
    font-size: 12px;
    color: #829ab1;
}

/* Logo */

.logo-box {
    text-align: center;
    margin-bottom: 10px;
}

.logo-box img {
    width: 62px;
    height: 62px;
    object-fit: contain;
    border-radius: 14px;
}

/* Buttons */

.stButton > button {
    border-radius: 10px;
    font-weight: 600;
}

/* Upload */

[data-testid="stFileUploader"] {
    background: #fbfdff;
    border-radius: 14px;
}

/* Tables */

.dataframe {
    border-radius: 10px;
}

/* Disclaimer */

.disclaimer {
    background: #fff8e7;
    border: 1px solid #f1d58a;
    border-radius: 12px;
    padding: 13px 16px;
    color: #6b4f00;
    font-size: 13px;
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE
# ============================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "current_result" not in st.session_state:
    st.session_state.current_result = None

if "page" not in st.session_state:
    st.session_state.page = "New Screening"

if "patient_counter" not in st.session_state:
    st.session_state.patient_counter = 1


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None

    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        return f"ERROR: {str(e)}"


model = load_model()


# ============================================================
# MODEL CLASSES
# ============================================================

CLASS_NAMES = [
    "No Diabetic Retinopathy",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative"
]


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image):
    image = image.convert("RGB")
    resized = image.resize((224, 224))

    arr = np.array(resized).astype(np.float32)

    # Standard normalization
    arr = arr / 255.0

    arr = np.expand_dims(arr, axis=0)

    return arr


# ============================================================
# FIND LAST CONVOLUTIONAL LAYER
# ============================================================

def find_last_conv_layer(model):

    for layer in reversed(model.layers):

        try:
            output_shape = layer.output.shape

            if len(output_shape) == 4:
                return layer.name

        except Exception:
            continue

    return None


# ============================================================
# GRAD-CAM
# ============================================================

def generate_gradcam(model, image_array, class_index):

    try:

        last_conv_name = find_last_conv_layer(model)

        if last_conv_name is None:
            return None

        last_conv_layer = model.get_layer(last_conv_name)

        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[
                last_conv_layer.output,
                model.output
            ]
        )

        with tf.GradientTape() as tape:

            conv_outputs, predictions = grad_model(image_array)

            if len(predictions.shape) == 2:
                class_score = predictions[:, class_index]
            else:
                class_score = predictions[class_index]

        grads = tape.gradient(class_score, conv_outputs)

        if grads is None:
            return None

        pooled_grads = tf.reduce_mean(
            grads,
            axis=(1, 2)
        )

        conv_outputs = conv_outputs[0]

        pooled_grads = pooled_grads[0]

        heatmap = tf.reduce_sum(
            conv_outputs * pooled_grads,
            axis=-1
        )

        heatmap = tf.maximum(heatmap, 0)

        max_value = tf.reduce_max(heatmap)

        if max_value > 0:
            heatmap /= max_value

        heatmap = heatmap.numpy()

        return heatmap

    except Exception:
        return None


# ============================================================
# CREATE HEATMAP IMAGE
# ============================================================

def create_heatmap_overlay(original_image, heatmap):

    if heatmap is None:
        return None

    original = original_image.convert("RGB")

    original = original.resize((224, 224))

    heatmap_img = Image.fromarray(
        np.uint8(255 * heatmap)
    ).resize((224, 224))

    # Use a simple red/yellow heatmap without requiring OpenCV
    heatmap_array = np.array(heatmap_img)

    # Create RGB heatmap
    rgb = np.zeros(
        (224, 224, 3),
        dtype=np.uint8
    )

    rgb[:, :, 0] = heatmap_array

    rgb[:, :, 1] = np.uint8(
        np.clip(heatmap_array.astype(float) * 0.45, 0, 255)
    )

    heatmap_rgb = Image.fromarray(rgb)

    overlay = Image.blend(
        original,
        heatmap_rgb,
        0.45
    )

    return overlay


# ============================================================
# PREDICTION
# ============================================================

def analyze_image(image):

    if model is None:
        return None, None, None, "Model file not found."

    if isinstance(model, str) and model.startswith("ERROR"):
        return None, None, None, model

    try:

        processed = preprocess_image(image)

        predictions = model.predict(
            processed,
            verbose=0
        )

        predictions = np.array(predictions)

        if predictions.ndim > 1:
            probabilities = predictions[0]
        else:
            probabilities = predictions

        # If model outputs logits, convert to probabilities
        if (
            np.min(probabilities) < 0
            or np.max(probabilities) > 1
            or abs(np.sum(probabilities) - 1) > 0.05
        ):
            probabilities = tf.nn.softmax(
                probabilities
            ).numpy()

        class_index = int(
            np.argmax(probabilities)
        )

        confidence = float(
            probabilities[class_index] * 100
        )

        if class_index < len(CLASS_NAMES):
            result = CLASS_NAMES[class_index]
        else:
            result = f"Class {class_index}"

        heatmap = generate_gradcam(
            model,
            processed,
            class_index
        )

        overlay = create_heatmap_overlay(
            image,
            heatmap
        )

        return (
            result,
            confidence,
            overlay,
            None
        )

    except Exception as e:

        return (
            None,
            None,
            None,
            str(e)
        )


# ============================================================
# PATIENT ID
# ============================================================

def generate_patient_id():

    current = st.session_state.patient_counter

    st.session_state.patient_counter += 1

    return f"P-{current:04d}"


# ============================================================
# SCREENING ID
# ============================================================

def generate_screening_id():

    return (
        "VG-"
        + datetime.now().strftime("%Y%m%d%H%M%S")
    )


# ============================================================
# RESULT CATEGORY
# ============================================================

def result_class(result):

    if result is None:
        return "result-warning"

    if result == "No Diabetic Retinopathy":
        return "result-normal"

    if result in ["Mild", "Moderate"]:
        return "result-warning"

    return "result-danger"


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    if os.path.exists(LOGO_PATH):

        st.markdown(
            '<div class="logo-box">',
            unsafe_allow_html=True
        )

        st.image(
            LOGO_PATH,
            width=65
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            "👁️",
            unsafe_allow_html=True
        )

    st.markdown(
        '<div class="sidebar-title">VisionGuard AI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-subtitle">'
        'AI-Powered Retinal Screening'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "New Screening",
            "Screening History",
            "Reports",
            "Settings"
        ],
        index=[
            "New Screening",
            "Screening History",
            "Reports",
            "Settings"
        ].index(st.session_state.page)
    )

    st.session_state.page = page

    st.divider()

    st.caption(
        "VisionGuard AI"
    )

    st.caption(
        "AI-assisted retinal image analysis"
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">VisionGuard AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-subtitle">'
    'Explainable AI workflow for retinal image analysis'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# NEW SCREENING
# ============================================================

if st.session_state.page == "New Screening":

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="card-title">Patient Information</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="card-subtitle">'
        'Enter patient details before image analysis.'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        patient_name = st.text_input(
            "Person Name *",
            placeholder="Enter patient name"
        )

    with col2:

        age = st.number_input(
            "Age *",
            min_value=1,
            max_value=120,
            value=40
        )

    with col3:

        gender = st.selectbox(
            "Gender",
            [
                "Male",
                "Female",
                "Other"
            ]
        )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # IMAGE UPLOAD
    # --------------------------------------------------------

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="card-title">Retinal Image</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="card-subtitle">'
        'Upload a clear fundus photograph for analysis.'
        '</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Upload retinal image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp"
        ],
        label_visibility="collapsed"
    )

    image = None

    if uploaded_file:

        try:

            image = Image.open(
                uploaded_file
            ).convert("RGB")

            st.image(
                image,
                caption="Selected retinal image",
                width=500
            )

        except Exception as e:

            st.error(
                f"Could not open image: {e}"
            )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # ANALYZE BUTTON
    # --------------------------------------------------------

    if st.button(
        "🔍 Analyze Retinal Image",
        type="primary",
        use_container_width=True
    ):

        if not patient_name.strip():

            st.error(
                "Please enter the patient name."
            )

        elif image is None:

            st.error(
                "Please upload a retinal image."
            )

        else:

            with st.spinner(
                "Analyzing retinal image..."
            ):

                result, confidence, gradcam, error = (
                    analyze_image(image)
                )

            if error:

                st.error(
                    "Analysis failed: " + error
                )

            else:

                patient_id = generate_patient_id()

                screening_id = generate_screening_id()

                screening_date = datetime.now()

                record = {
                    "Patient ID": patient_id,
                    "Screening ID": screening_id,
                    "Name": patient_name.strip(),
                    "Age": age,
                    "Gender": gender,
                    "Result": result,
                    "Confidence": round(confidence, 2),
                    "Date": screening_date.strftime(
                        "%Y-%m-%d"
                    ),
                    "Time": screening_date.strftime(
                        "%H:%M"
                    )
                }

                st.session_state.history.append(
                    record
                )

                st.session_state.current_result = {
                    "patient_id": patient_id,
                    "screening_id": screening_id,
                    "name": patient_name.strip(),
                    "age": age,
                    "gender": gender,
                    "result": result,
                    "confidence": confidence,
                    "image": image,
                    "gradcam": gradcam,
                    "date": screening_date
                }

                st.success(
                    "Analysis completed successfully."
                )

    # --------------------------------------------------------
    # CURRENT RESULT
    # --------------------------------------------------------

    current = st.session_state.current_result

    if current:

        st.markdown(
            '<div class="card">',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="card-title">'
            'AI Interpretation'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="card-subtitle">'
            'Model prediction and explainability view'
            '</div>',
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)

        with col1:

            st.image(
                current["image"],
                caption="Retinal image",
                use_container_width=True
            )

        with col2:

            if current["gradcam"] is not None:

                st.image(
                    current["gradcam"],
                    caption="Grad-CAM attention map",
                    use_container_width=True
                )

            else:

                st.info(
                    "Grad-CAM could not be generated "
                    "for this model."
                )

        st.markdown(
            '<br>',
            unsafe_allow_html=True
        )

        rclass = result_class(
            current["result"]
        )

        st.markdown(
            f"""
            <div class="{rclass}">
                <div class="result-title">
                    {current["result"]}
                </div>
                <div>
                    Model confidence:
                    <b>{current["confidence"]:.2f}%</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            '<br>',
            unsafe_allow_html=True
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value">
                    {current["patient_id"]}
                    </div>
                    <div class="metric-label">
                    Patient ID
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with c2:

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value">
                    {current["age"]}
                    </div>
                    <div class="metric-label">
                    Age
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with c3:

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value">
                    {current["confidence"]:.1f}%
                    </div>
                    <div class="metric-label">
                    Confidence
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with c4:

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value">
                    {current["screening_id"]}
                    </div>
                    <div class="metric-label">
                    Screening ID
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown(
            """
            <div class="disclaimer">
            This AI output is intended to assist retinal image
            screening and research workflows. It should not be
            used as a substitute for professional medical
            diagnosis or clinical judgment.
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


# ============================================================
# SCREENING HISTORY
# ============================================================

elif st.session_state.page == "Screening History":

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="card-title">Screening History</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="card-subtitle">'
        'Search and filter previous screening records.'
        '</div>',
        unsafe_allow_html=True
    )

    if len(st.session_state.history) == 0:

        st.info(
            "No screening records available yet."
        )

    else:

        search = st.text_input(
            "Search patient by name or ID",
            placeholder="Enter name, Patient ID or Screening ID"
        )

        c1, c2 = st.columns(2)

        with c1:

            start_date = st.date_input(
                "From date",
                value=date.today().replace(
                    day=1
                )
            )

        with c2:

            end_date = st.date_input(
                "To date",
                value=date.today()
            )

        df = pd.DataFrame(
            st.session_state.history
        )

        df["DateObject"] = pd.to_datetime(
            df["Date"]
        ).dt.date

        if search:

            search_lower = search.lower()

            df = df[
                df["Name"].str.lower().str.contains(
                    search_lower,
                    na=False
                )
                |
                df["Patient ID"].str.lower().str.contains(
                    search_lower,
                    na=False
                )
                |
                df["Screening ID"].str.lower().str.contains(
                    search_lower,
                    na=False
                )
            ]

        df = df[
            (df["DateObject"] >= start_date)
            &
            (df["DateObject"] <= end_date)
        ]

        display_columns = [
            "Patient ID",
            "Screening ID",
            "Name",
            "Age",
            "Gender",
            "Result",
            "Confidence",
            "Date",
            "Time"
        ]

        st.dataframe(
            df[display_columns],
            use_container_width=True,
            hide_index=True
        )

        st.write(
            f"Showing **{len(df)}** record(s)."
        )

        st.divider()

        if st.button(
            "🗑️ Clear All History",
            type="secondary"
        ):

            st.session_state.history = []

            st.session_state.current_result = None

            st.success(
                "Screening history cleared."
            )

            st.rerun()

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


# ============================================================
# REPORTS
# ============================================================

elif st.session_state.page == "Reports":

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="card-title">Reports</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="card-subtitle">'
        'View and export screening records.'
        '</div>',
        unsafe_allow_html=True
    )

    if len(st.session_state.history) == 0:

        st.info(
            "No screening records available for reports."
        )

    else:

        df = pd.DataFrame(
            st.session_state.history
        )

        st.metric(
            "Total Screenings",
            len(df)
        )

        st.markdown("### Screening Summary")

        summary = (
            df["Result"]
            .value_counts()
            .reset_index()
        )

        summary.columns = [
            "Category",
            "Count"
        ]

        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True
        )

        st.markdown("### All Screening Records")

        report_columns = [
            "Patient ID",
            "Screening ID",
            "Name",
            "Age",
            "Gender",
            "Result",
            "Confidence",
            "Date",
            "Time"
        ]

        st.dataframe(
            df[report_columns],
            use_container_width=True,
            hide_index=True
        )

        csv_data = df[
            report_columns
        ].to_csv(
            index=False
        )

        st.download_button(
            "⬇️ Download Report (CSV)",
            data=csv_data,
            file_name="visionguard_screening_report.csv",
            mime="text/csv",
            use_container_width=True
        )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


# ============================================================
# SETTINGS
# ============================================================

elif st.session_state.page == "Settings":

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="card-title">Settings</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="card-subtitle">'
        'Application and model information.'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("### Model")

    if model is None:

        st.error(
            "Model file not found: "
            + MODEL_PATH
        )

    elif isinstance(model, str):

        st.error(model)

    else:

        st.success(
            "VisionGuard CNN model loaded successfully."
        )

        try:

            st.write(
                "Model input shape:",
                model.input_shape
            )

            st.write(
                "Model output shape:",
                model.output_shape
            )

        except Exception:
            pass

    st.divider()

    st.markdown("### Application")

    st.write(
        "**Application:** VisionGuard AI"
    )

    st.write(
        "**Purpose:** AI-assisted retinal image screening"
    )

    st.write(
        "**Input size:** 224 × 224 pixels"
    )

    st.write(
        "**Explainability:** Grad-CAM when supported by the model"
    )

    st.divider()

    st.markdown("### Data")

    st.write(
        "Screening history is stored in the current "
        "application session."
    )

    st.warning(
        "For a production system, patient records should "
        "be stored in a secure database with appropriate "
        "privacy and access controls."
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div style="
        text-align:center;
        color:#829ab1;
        font-size:12px;
        padding:30px 0 10px 0;
    ">
        VisionGuard AI · AI-assisted retinal screening
    </div>
    """,
    unsafe_allow_html=True
)
