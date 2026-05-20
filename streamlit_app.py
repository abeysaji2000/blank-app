import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import tempfile
import os
import time
import matplotlib.pyplot as plt

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="NeuroVision AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# PROFESSIONAL UI
# =========================================================
st.markdown("""
<style>

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0B1020 0%, #111827 100%);
    color: #E5E7EB;
}

/* HEADER */
.main-header {
    padding-top: 10px;
    padding-bottom: 20px;
}

.title {
    font-size: 52px;
    font-weight: 700;
    color: #F8FAFC;
}

.subtitle {
    color: #94A3B8;
    font-size: 18px;
    margin-top: 6px;
}

/* CARDS */
.card {
    background: rgba(17, 24, 39, 0.82);
    border-radius: 22px;
    padding: 24px;
    border: 1px solid rgba(255,255,255,0.05);
    backdrop-filter: blur(12px);
    box-shadow: 0 10px 30px rgba(0,0,0,0.25);
}

/* SECTION */
.section-title {
    color: #F8FAFC;
    font-size: 28px;
    font-weight: 600;
    margin-bottom: 16px;
}

/* METRIC */
.metric-card {
    background: linear-gradient(
        145deg,
        rgba(15,23,42,0.95),
        rgba(17,24,39,0.95)
    );

    border-radius: 18px;
    padding: 20px;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.05);
}

.metric-value {
    color: #F8FAFC;
    font-size: 32px;
    font-weight: 700;
}

.metric-label {
    color: #94A3B8;
    font-size: 14px;
    margin-top: 8px;
}

/* UPLOAD */
.upload-box {
    border: 2px dashed rgba(148,163,184,0.25);
    border-radius: 24px;
    padding: 35px;
    text-align: center;
    background: rgba(15,23,42,0.4);
}

/* BUTTON */
.stButton>button {
    background: linear-gradient(135deg, #2563EB, #1D4ED8);
    color: white;
    border: none;
    border-radius: 14px;
    padding: 12px 20px;
    font-weight: 600;
}

.stButton>button:hover {
    background: linear-gradient(135deg, #1D4ED8, #1E40AF);
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background-color: #0F172A;
}

/* HIDE STREAMLIT */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================
st.markdown("""
<div class="main-header">

<div class="title">
NeuroVision AI
</div>

<div class="subtitle">
Comparative Brain Metastases Segmentation Platform
</div>

</div>
""", unsafe_allow_html=True)

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.markdown("## AI Configuration")

confidence = st.sidebar.slider(
    "Detection Confidence",
    0.10,
    1.00,
    0.25,
    0.05
)

img_size = st.sidebar.selectbox(
    "Inference Resolution",
    [256, 512, 640, 768, 1024],
    index=2
)

opacity = st.sidebar.slider(
    "Overlay Transparency",
    0.1,
    1.0,
    0.30,
    0.05
)

border_thickness = st.sidebar.slider(
    "Contour Thickness",
    1,
    10,
    2
)

st.sidebar.markdown("---")

st.sidebar.markdown("""
### Loaded Models

| Model | Description |
|---|---|
| segrun1.pt | BraTS Dataset Model |
| Sam model.pt | SAM Annotation Model |

""")

# =========================================================
# LOAD MODELS
# =========================================================
MODEL_1_PATH = "segrun1.pt"
MODEL_2_PATH = "Sam model.pt"

@st.cache_resource
def load_models():

    model1 = YOLO(MODEL_1_PATH)
    model2 = YOLO(MODEL_2_PATH)

    return model1, model2

try:
    model_dataset, model_sam = load_models()

except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# =========================================================
# UPLOAD SECTION
# =========================================================
st.markdown("""
<div class="upload-box">

<h2>Upload MRI Scan</h2>

Compare segmentation outputs between two AI models

</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "",
    type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"]
)

# =========================================================
# SEGMENTATION FUNCTION
# =========================================================
def generate_segmentation(model, image_path, image_np):

    start_time = time.time()

    results = model.predict(
        source=image_path,
        conf=confidence,
        imgsz=img_size,
        save=False,
        verbose=False
    )

    inference_time = round(time.time() - start_time, 2)

    result = results[0]

    segmented = image_np.copy()
    overlay = segmented.copy()

    total_regions = 0
    total_area = 0

    if result.masks is not None:

        masks = result.masks.data.cpu().numpy()

        total_regions = len(masks)

        for mask in masks:

            # Resize
            mask = cv2.resize(
                mask,
                (segmented.shape[1], segmented.shape[0])
            )

            # Binary
            mask = (mask > 0.5).astype(np.uint8)

            area = np.sum(mask)
            total_area += area

            contours, _ = cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            # Transparent fill
            cv2.fillPoly(
                overlay,
                contours,
                color=(0, 140, 255)
            )

            # Polygon contour
            cv2.drawContours(
                segmented,
                contours,
                -1,
                (255, 255, 255),
                border_thickness
            )

    segmented = cv2.addWeighted(
        overlay,
        opacity,
        segmented,
        1 - opacity,
        0
    )

    return segmented, total_regions, total_area, inference_time

# =========================================================
# PROCESS IMAGE
# =========================================================
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    # Save temp image
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:

        temp_path = tmp.name
        image.save(temp_path)

    # =====================================================
    # RUN BOTH MODELS
    # =====================================================
    with st.spinner("Running comparative segmentation..."):

        seg1, regions1, area1, time1 = generate_segmentation(
            model_dataset,
            temp_path,
            image_np
        )

        seg2, regions2, area2, time2 = generate_segmentation(
            model_sam,
            temp_path,
            image_np
        )

    # =====================================================
    # ORIGINAL MRI
    # =====================================================
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
    <div class="section-title">
    Original MRI Scan
    </div>
    """, unsafe_allow_html=True)

    st.image(image_np, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # =====================================================
    # OUTPUTS
    # =====================================================
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    # MODEL 1
    with col1:

        st.markdown("""
        <div class="card">
        <div class="section-title">
        BraTS Dataset Model
        </div>
        """, unsafe_allow_html=True)

        st.image(seg1, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # MODEL 2
    with col2:

        st.markdown("""
        <div class="card">
        <div class="section-title">
        SAM Annotation Model
        </div>
        """, unsafe_allow_html=True)

        st.image(seg2, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # =====================================================
    # METRICS
    # =====================================================
    st.markdown("<br>", unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)

    with m1:

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{regions1}</div>
            <div class="metric-label">
            Dataset Regions
            </div>
        </div>
        """, unsafe_allow_html=True)

    with m2:

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{regions2}</div>
            <div class="metric-label">
            SAM Regions
            </div>
        </div>
        """, unsafe_allow_html=True)

    with m3:

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{time1}s</div>
            <div class="metric-label">
            Dataset Model Time
            </div>
        </div>
        """, unsafe_allow_html=True)

    with m4:

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{time2}s</div>
            <div class="metric-label">
            SAM Model Time
            </div>
        </div>
        """, unsafe_allow_html=True)

    # =====================================================
    # GRAPHS
    # =====================================================
    st.markdown("<br><br>", unsafe_allow_html=True)

    g1, g2 = st.columns(2)

    # -----------------------------------------------------
    # REGION GRAPH
    # -----------------------------------------------------
    with g1:

        st.markdown("""
        <div class="card">
        <div class="section-title">
        Segmentation Region Comparison
        </div>
        """, unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(6,4))

        models = [
            "Dataset Model",
            "SAM Model"
        ]

        region_values = [
            regions1,
            regions2
        ]

        ax.bar(models, region_values)

        ax.set_ylabel("Detected Regions")

        st.pyplot(fig)

        st.markdown("</div>", unsafe_allow_html=True)

    # -----------------------------------------------------
    # AREA GRAPH
    # -----------------------------------------------------
    with g2:

        st.markdown("""
        <div class="card">
        <div class="section-title">
        Segmentation Area Comparison
        </div>
        """, unsafe_allow_html=True)

        fig2, ax2 = plt.subplots(figsize=(6,4))

        area_values = [
            area1,
            area2
        ]

        ax2.plot(
            models,
            area_values,
            linewidth=3,
            marker='o'
        )

        ax2.set_ylabel("Segmented Area")

        st.pyplot(fig2)

        st.markdown("</div>", unsafe_allow_html=True)

    # =====================================================
    # AI INTERPRETATION
    # =====================================================
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
    <div class="section-title">
    Comparative AI Interpretation
    </div>
    """, unsafe_allow_html=True)

    if regions1 > regions2:

        st.success("""
        The BraTS dataset-trained model identified more segmented
        metastatic structures than the SAM annotation model.
        """)

    elif regions2 > regions1:

        st.success("""
        The SAM annotation model identified more segmented
        metastatic structures than the BraTS dataset-trained model.
        """)

    else:

        st.info("""
        Both AI models produced similar segmentation counts.
        """)

    st.markdown("</div>", unsafe_allow_html=True)

    # =====================================================
    # DOWNLOADS
    # =====================================================
    st.markdown("<br>", unsafe_allow_html=True)

    d1, d2 = st.columns(2)

    with d1:

        seg1_pil = Image.fromarray(seg1)

        temp1 = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".png"
        )

        seg1_pil.save(temp1.name)

        with open(temp1.name, "rb") as f:

            st.download_button(
                label="Download Dataset Model Output",
                data=f,
                file_name="dataset_model_output.png",
                mime="image/png"
            )

    with d2:

        seg2_pil = Image.fromarray(seg2)

        temp2 = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".png"
        )

        seg2_pil.save(temp2.name)

        with open(temp2.name, "rb") as f:

            st.download_button(
                label="Download SAM Model Output",
                data=f,
                file_name="sam_model_output.png",
                mime="image/png"
            )

    # Cleanup
    os.remove(temp_path)

# =========================================================
# FOOTER
# =========================================================
st.markdown("<br><br>", unsafe_allow_html=True)

st.markdown("""
<div style='text-align:center; color:#64748B; font-size:14px;'>

NeuroVision AI • Comparative Brain Metastases Segmentation Platform

</div>
""", unsafe_allow_html=True)