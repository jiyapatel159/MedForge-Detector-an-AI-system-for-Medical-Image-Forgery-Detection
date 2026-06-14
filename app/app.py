import os
import sys
import torch
import numpy as np
import cv2
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image, ImageChops, ImageEnhance
from torchvision import transforms
import pydicom
import io

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

from src.models.cnn_baseline import MedicalForgeryDetectorCNN

st.set_page_config(
    page_title = "MedForge Detector",
    page_icon  = "🔬",
    layout     = "wide",
)

st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem; font-weight: 700;
        color: #1a1a2e; text-align: center;
        padding: 1rem 0 0.2rem;
    }
    .sub-title {
        font-size: 1rem; color: #555;
        text-align: center; margin-bottom: 2rem;
    }
    .result-real {
        background: #d4edda; border-left: 5px solid #28a745;
        padding: 1rem 1.5rem; border-radius: 8px;
        font-size: 1.3rem; font-weight: 600; color: #155724;
    }
    .result-forged {
        background: #f8d7da; border-left: 5px solid #dc3545;
        padding: 1rem 1.5rem; border-radius: 8px;
        font-size: 1.3rem; font-weight: 600; color: #721c24;
    }
    .dicom-box {
        background: #e8f4fd; border: 1px solid #bee5eb;
        border-radius: 8px; padding: 1rem;
        font-family: monospace; font-size: 0.85rem;
    }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
@st.cache_resource
def load_model():
    """Load CNN baseline model — cached so it loads only once"""
    device    = torch.device("cpu")
    model     = MedicalForgeryDetectorCNN()
    ckpt_path = "models_saved/cnn_baseline_best.pth"

    if not os.path.exists(ckpt_path):
        return None, None

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    val_acc = f"{ckpt['val_acc']*100:.1f}%"
    return model, val_acc


def generate_ela(pil_image, quality=90):
    """Generate Error Level Analysis map"""
    original   = pil_image.convert('RGB')
    buf        = io.BytesIO()
    original.save(buf, format='JPEG', quality=quality)
    buf.seek(0)
    compressed = Image.open(buf).convert('RGB')
    ela        = ImageChops.difference(original, compressed)
    extrema    = ela.getextrema()
    max_diff   = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    ela = ImageEnhance.Brightness(ela).enhance(255.0 / max_diff)
    return ela


def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]
        )
    ])


def predict(model, pil_image):
    """Run CNN baseline prediction — image only, no ELA"""
    transform  = get_transform()
    ela        = generate_ela(pil_image)
    img_tensor = transform(pil_image.convert('RGB')).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)
        prob   = output.item()

    return prob, prob > 0.5, ela


def generate_gradcam(model, pil_image):
    """
    Grad-CAM for CNN baseline.
    Auto-detects correct layer name from your model.
    """
    transform  = get_transform()
    img_tensor = transform(pil_image.convert('RGB')).unsqueeze(0)

    activations = {}
    gradients   = {}

    # ── Auto-detect correct conv layer ────────────────────
    # Tries multiple possible attribute names
    target = None

    try:
        # Option 1: block5 has .block attribute with Conv2d at [0]
        target = model.block5.block[0]
    except AttributeError:
        pass

    if target is None:
        try:
            # Option 2: block5 has .conv attribute directly
            target = model.block5.conv
        except AttributeError:
            pass

    if target is None:
        try:
            # Option 3: walk all Conv2d layers, pick last one
            all_convs = [
                m for m in model.modules()
                if isinstance(m, torch.nn.Conv2d)
            ]
            target = all_convs[-1]
        except Exception:
            pass

    if target is None:
        raise ValueError("Could not find any Conv2d layer in model")

    # ── Register hooks ────────────────────────────────────
    def fwd_hook(m, i, o):
        activations["feat"] = o.clone().detach()

    def bwd_hook(m, gi, go):
        gradients["grad"] = go[0].clone().detach()

    h1 = target.register_forward_hook(fwd_hook)
    h2 = target.register_full_backward_hook(bwd_hook)

    try:
        model.zero_grad()
        img_in = img_tensor.clone().detach().requires_grad_(True)
        out    = model(img_in)
        out.backward()

        if "grad" not in gradients or "feat" not in activations:
            raise ValueError("Hooks did not capture data")

        grads   = gradients["grad"][0]       # (C, H, W)
        acts    = activations["feat"][0]     # (C, H, W)
        weights = grads.mean(dim=(1, 2))     # (C,)

        cam = torch.zeros(acts.shape[1:])
        for i, w in enumerate(weights):
            cam += w * acts[i]

        cam = F.relu(cam).numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        cam = cv2.resize(cam, (224, 224))

    finally:
        h1.remove()
        h2.remove()

    return cam


def create_overlay(original_pil, heatmap):
    """Blend heatmap onto original image"""
    orig_np       = np.array(
        original_pil.convert('RGB').resize((224, 224))
    )
    heatmap_u8    = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay       = cv2.addWeighted(
        orig_np, 0.55, heatmap_color, 0.45, 0
    )
    return Image.fromarray(overlay)


def load_dicom(uploaded_file):
    """Read DICOM file → PIL image + metadata dict"""
    import tempfile
    with tempfile.NamedTemporaryFile(
            suffix='.dcm', delete=False) as f:
        f.write(uploaded_file.read())
        tmp_path = f.name

    dcm = pydicom.dcmread(tmp_path)
    arr = dcm.pixel_array.astype(float)
    arr = (arr - arr.min()) / (arr.max() - arr.min()) * 255
    img = Image.fromarray(arr.astype(np.uint8)).convert('RGB')

    meta = {}
    for tag in ['PatientID', 'PatientName', 'Modality',
                'StudyDate', 'InstitutionName',
                'Manufacturer', 'StudyDescription']:
        try:
            meta[tag] = str(getattr(dcm, tag, 'N/A'))
        except Exception:
            meta[tag] = 'N/A'

    os.unlink(tmp_path)
    return img, meta


def confidence_bar(prob, is_forged):
    """Visual confidence bar using matplotlib"""
    fig, ax = plt.subplots(figsize=(5, 0.6))
    fig.patch.set_alpha(0)
    ax.set_facecolor('none')
    color = '#dc3545' if is_forged else '#28a745'
    val   = prob if is_forged else 1 - prob
    ax.barh(0, 1,   height=0.5, color="#12467a")
    ax.barh(0, val, height=0.5, color=color)
    ax.set_xlim(0, 1)
    ax.axis('off')
    plt.tight_layout(pad=0)
    return fig


# ══════════════════════════════════════════════════════════════
def main():

    # ── Header ────────────────────────────────────────────
    st.markdown(
        '<div class="main-title">🔬 MedForge Detector</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="sub-title">Medical Image Forgery Detection '
        '— CNN Deep Learning Model</div>',
        unsafe_allow_html=True
    )

    # ── Load model ────────────────────────────────────────
    with st.spinner("Loading model..."):
        model, val_acc = load_model()

    if model is None:
        st.error("Model not found at models_saved/cnn_baseline_best.pth")
        st.info("Run: python src/training/train.py")
        return

    # ── Sidebar ───────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Model Info")
        st.success(f"CNN Baseline — Val accuracy: {val_acc}")
        st.info(
            "**Architecture:** 5-block CNN from scratch\n\n"
            "**Parameters:** 1,734,689\n\n"
            "**Test accuracy:** 93.04%\n\n"
            "**AUC-ROC:** 0.9809"
        )

        st.markdown("---")
        st.header("📂 Upload Image")
        uploaded = st.file_uploader(
            "Choose a medical image",
            type=["png", "jpg", "jpeg", "dcm"],
            help="Supports PNG, JPG and DICOM (.dcm)"
        )

        st.markdown("---")
        show_ela     = st.checkbox("Show ELA Map",        value=True)
        show_gradcam = st.checkbox("Show Grad-CAM",       value=True)
        show_dicom   = st.checkbox("Show DICOM Metadata", value=True)

        st.markdown("---")
        st.markdown("**Detects:**")
        st.markdown(
            "- Copy-move forgery\n"
            "- Image splicing\n"
            "- Content removal\n"
            "- Text addition"
        )

    # ── Landing page ──────────────────────────────────────
    if uploaded is None:
        st.markdown("### How it works")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.info("**1. Upload**\nUpload any X-Ray or MRI image")
        with c2:
            st.info("**2. Preprocess**\nImage resized and normalized")
        with c3:
            st.info("**3. CNN Analysis**\n5-block CNN scans for artifacts")
        with c4:
            st.info("**4. Result**\nReal/Forged + Grad-CAM heatmap")

        st.markdown("---")
        st.markdown("### Model Performance")
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Test Accuracy",  "93.04%")
        mc2.metric("Precision",      "94.42%")
        mc3.metric("Recall",         "94.88%")
        mc4.metric("AUC-ROC",        "0.9809")
        return

    # ── Process uploaded file ─────────────────────────────
    dicom_meta = None
    if uploaded.name.lower().endswith('.dcm'):
        with st.spinner("Reading DICOM..."):
            pil_image, dicom_meta = load_dicom(uploaded)
    else:
        pil_image = Image.open(uploaded).convert('RGB')

    # Predict
    with st.spinner("Analysing image..."):
        prob, is_forged, ela_image = predict(model, pil_image)

    # Grad-CAM
    gradcam_overlay = None
    if show_gradcam:
        with st.spinner("Generating Grad-CAM..."):
            try:
                heatmap         = generate_gradcam(model, pil_image)
                gradcam_overlay = create_overlay(pil_image, heatmap)
            except Exception as e:
                st.warning(f"Grad-CAM unavailable: {e}")

    # ── Result banner ─────────────────────────────────────
    st.markdown("---")
    if is_forged:
        st.markdown(
            f'<div class="result-forged">'
            f'⚠️ FORGED IMAGE DETECTED &nbsp;|&nbsp; '
            f'Confidence: {prob*100:.1f}%'
            f'</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="result-real">'
            f'✅ AUTHENTIC IMAGE &nbsp;|&nbsp; '
            f'Confidence: {(1-prob)*100:.1f}%'
            f'</div>',
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Confidence bar
    st.markdown("**Forgery probability**")
    col_bar, _ = st.columns([3, 1])
    with col_bar:
        st.pyplot(
            confidence_bar(prob, is_forged),
            use_container_width=False
        )
        st.caption(
            f"Forgery score: {prob*100:.2f}%  |  "
            f"Authentic score: {(1-prob)*100:.2f}%"
        )

    st.markdown("---")

    # ── Image columns ─────────────────────────────────────
    panels = [
        ("Original Image", pil_image.resize((224, 224)),
         "Uploaded medical image")
    ]
    if show_ela:
        panels.append((
            "ELA Map",
            ela_image.resize((224, 224)),
            "Bright regions = suspicious compression anomaly"
        ))
    if show_gradcam and gradcam_overlay is not None:
        panels.append((
            "Grad-CAM Heatmap",
            gradcam_overlay,
            "Red = region model flagged as tampered"
        ))

    cols = st.columns(len(panels))
    for col, (title, img, caption) in zip(cols, panels):
        with col:
            st.subheader(title)
            st.image(img, width=224)
            st.caption(caption)

    # ── Metrics row ───────────────────────────────────────
    st.markdown("---")
    st.subheader("📊 Detection Metrics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Prediction",
              "FORGED" if is_forged else "REAL")
    m2.metric("Forgery Score",   f"{prob*100:.2f}%")
    m3.metric("Authentic Score", f"{(1-prob)*100:.2f}%")
    m4.metric("Model",           "CNN Baseline")

    # ── DICOM metadata ────────────────────────────────────
    if dicom_meta and show_dicom:
        st.markdown("---")
        st.subheader("🏥 DICOM Metadata")
        dc1, dc2 = st.columns(2)
        items = list(dicom_meta.items())
        half  = len(items) // 2
        with dc1:
            for k, v in items[:half]:
                st.markdown(
                    f'<div class="dicom-box">'
                    f'<b>{k}:</b> {v}</div><br>',
                    unsafe_allow_html=True
                )
        with dc2:
            for k, v in items[half:]:
                st.markdown(
                    f'<div class="dicom-box">'
                    f'<b>{k}:</b> {v}</div><br>',
                    unsafe_allow_html=True
                )

    # ── Footer ────────────────────────────────────────────
    st.markdown("---")
    st.caption(
        "MedForge Detector | CNN Baseline | "
        "Test Accuracy 93.04% · AUC-ROC 0.9809 · F1 94.65%"
    )


if __name__ == "__main__":
    main()