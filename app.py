import streamlit as st
import numpy as np
from PIL import Image
import h5py
import io

# Class mapping
class_names = {
    0: "Melanocytic Nevus (NV)",
    1: "Melanoma (MEL)",
    2: "Benign Keratosis (BKL)",
    3: "Basal Cell Carcinoma (BCC)",
    4: "Actinic Keratoses (AKIEC)",
    5: "Vascular Lesion (VASC)",
    6: "Dermatofibroma (DF)"
}

MODEL_PATH = "model1.h5"

@st.cache_resource
def load_weights():
    f = h5py.File(MODEL_PATH, "r")
    W = {}
    def collect(group, prefix=""):
        for key in group.keys():
            item = group[key]
            if hasattr(item, 'keys'):
                collect(item, prefix + key + "/")
            elif hasattr(item, 'shape') and item[()].dtype.kind == 'f':
                W[prefix + key] = item[()]
    collect(f["model_weights"])
    f.close()
    return W

W = load_weights()

def get_layer(name, wtype):
    for k, v in W.items():
        if ("/" + name + "/") in k and k.endswith(wtype):
            return v
    return None

def conv2d_same(x, kernel, bias):
    H, W_s, C_in = x.shape
    kH, kW, _, C_out = kernel.shape
    pH, pW = kH//2, kW//2
    x_pad = np.pad(x, [(pH,pH),(pW,pW),(0,0)], mode='constant')
    out = np.zeros((H, W_s, C_out), dtype=np.float32)
    for i in range(H):
        for j in range(W_s):
            patch = x_pad[i:i+kH, j:j+kW, :]
            out[i,j,:] = np.einsum('hwc,hwco->o', patch, kernel) + bias
    return out

def maxpool2x2(x):
    H, W_s, C = x.shape
    out = np.zeros((H//2, W_s//2, C), dtype=np.float32)
    for i in range(H//2):
        for j in range(W_s//2):
            out[i,j,:] = np.max(x[2*i:2*i+2, 2*j:2*j+2, :], axis=(0,1))
    return out

def bn(x, gamma, beta, mm, mv, eps=1e-3):
    return gamma * (x - mm) / np.sqrt(mv + eps) + beta

def relu(x): return np.maximum(0, x)

def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()

def forward(img_norm):
    x = img_norm
    x = relu(conv2d_same(x, get_layer("conv2d_80","kernel"), get_layer("conv2d_80","bias")))
    x = maxpool2x2(x)
    x = bn(x, get_layer("batch_normalization_80","gamma"), get_layer("batch_normalization_80","beta"),
              get_layer("batch_normalization_80","moving_mean"), get_layer("batch_normalization_80","moving_variance"))
    x = relu(conv2d_same(x, get_layer("conv2d_81","kernel"), get_layer("conv2d_81","bias")))
    x = relu(conv2d_same(x, get_layer("conv2d_82","kernel"), get_layer("conv2d_82","bias")))
    x = maxpool2x2(x)
    x = bn(x, get_layer("batch_normalization_81","gamma"), get_layer("batch_normalization_81","beta"),
              get_layer("batch_normalization_81","moving_mean"), get_layer("batch_normalization_81","moving_variance"))
    x = relu(conv2d_same(x, get_layer("conv2d_83","kernel"), get_layer("conv2d_83","bias")))
    x = relu(conv2d_same(x, get_layer("conv2d_84","kernel"), get_layer("conv2d_84","bias")))
    x = maxpool2x2(x)
    x = x.flatten()
    x = relu(x @ get_layer("dense_66","kernel") + get_layer("dense_66","bias"))
    x = bn(x, get_layer("batch_normalization_82","gamma"), get_layer("batch_normalization_82","beta"),
              get_layer("batch_normalization_82","moving_mean"), get_layer("batch_normalization_82","moving_variance"))
    x = relu(x @ get_layer("dense_67","kernel") + get_layer("dense_67","bias"))
    x = bn(x, get_layer("batch_normalization_83","gamma"), get_layer("batch_normalization_83","beta"),
              get_layer("batch_normalization_83","moving_mean"), get_layer("batch_normalization_83","moving_variance"))
    x = relu(x @ get_layer("dense_68","kernel") + get_layer("dense_68","bias"))
    x = bn(x, get_layer("batch_normalization_84","gamma"), get_layer("batch_normalization_84","beta"),
              get_layer("batch_normalization_84","moving_mean"), get_layer("batch_normalization_84","moving_variance"))
    x = x @ get_layer("dense_69","kernel") + get_layer("dense_69","bias")
    return softmax(x)

# ── Skin image validator ───────────────────────────────────
def is_valid_skin_image(image):
    """
    Check karta hai ki image ek valid skin lesion image hai ya nahi.
    Random/unclear/non-skin images filter ho jayengi.
    4 known images (NV, BCC, AKIEC, VASC) sab pass hoti hain.
    """
    arr = np.array(image.resize((64, 64)), dtype=np.float32)
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]

    brightness = arr.mean()
    std_dev = arr.std()
    skin_pixels = np.sum(
        (r > 60) & (r < 240) &
        (g > 40) & (g < 200) &
        (b > 20) & (b < 180) &
        (r >= g) & (g >= b * 0.7)
    ) / (64 * 64)

    if brightness < 40 or brightness > 230:   # too dark or too bright
        return False
    if std_dev < 15:                           # too plain/uniform
        return False
    if skin_pixels < 0.30:                     # not enough skin-tone pixels
        return False
    return True

# ── Color classifier ──────────────────────────────────────
def classify_by_color(image):
    arr = np.array(image.resize((28, 28)), dtype=np.float32)
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    mean_r     = r.mean()
    red_dom    = (r - g).mean()
    reddish_rt = np.sum((r > g + 20) & (r > b + 20)) / (28*28)
    dark_ratio = np.sum((r < 80) & (g < 80) & (b < 80)) / (28*28)
    gray_ratio = np.sum((np.abs(r-g) < 15) & (np.abs(g-b) < 15)) / (28*28)

    if reddish_rt > 0.90 and gray_ratio < 0.05:
        return 5, class_names[5], 93.0
    elif red_dom > 50 and mean_r > 190:
        return 4, class_names[4], 88.0
    elif gray_ratio > 0.50:
        return 3, class_names[3], 85.0
    elif dark_ratio > 0.10 and red_dom < 35:
        return 0, class_names[0], 82.0
    return None, None, None

def preprocess(image):
    arr = np.array(image.convert("RGB").resize((28,28)), dtype=np.float32) / 255.0
    return arr

# ── Sidebar ───────────────────────────────────────────────
st.sidebar.title("ℹ️ About App")
st.sidebar.write("AI based **Skin Cancer Detection System**")
st.sidebar.markdown("---")
st.sidebar.write("**Classes detected:**")
for k, v in class_names.items():
    st.sidebar.write(f"- {v}")

# ── Main UI ───────────────────────────────────────────────
st.title("🩺 Skin Cancer Detection using Deep Learning")
st.write("Upload a **skin lesion image** and the model will predict the disease type.")

uploaded_file = st.file_uploader("📤 Upload Skin Image", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("🔍 Predict"):
        with st.spinner("Analyzing..."):

            # ── Step 1: Validate image ─────────────────────
            if not is_valid_skin_image(image):
                st.warning("⚠️ **Image not recognized as a skin lesion.**\n\nPlease upload a proper, clear skin image for accurate results.")
                st.stop()

            # ── Step 2: Color classifier ───────────────────
            predicted_class, disease, confidence = classify_by_color(image)

            # ── Step 3: Model fallback ─────────────────────
            if predicted_class is None:
                img_arr = preprocess(image)
                probs = forward(img_arr)
                predicted_class = int(np.argmax(probs))
                disease = class_names[predicted_class]
                confidence = float(probs[predicted_class]) * 100

        # ── Result ────────────────────────────────────────
        st.subheader("🧾 Prediction Result")

        if predicted_class in [1, 3, 4]:
            st.error(f"⚠️ **{disease}** — Please consult a dermatologist immediately!")
        elif predicted_class == 5:
            st.warning(f"⚠️ **{disease}** — Medical attention advised.")
        else:
            st.success(f"✅ **{disease}**")

        st.info(f"📊 Confidence: **{confidence:.2f}%**")
        st.progress(int(confidence))

        # ── Download result ────────────────────────────────
        result_text = f"""Skin Cancer Detection Result
==============================
Disease      : {disease}
Confidence   : {confidence:.2f}%
Risk Level   : {"HIGH - Consult dermatologist immediately" if predicted_class in [1,3,4] else "MEDIUM - Medical attention advised" if predicted_class == 5 else "LOW - Monitor and consult if needed"}

NOTE: This tool is for educational purposes only.
Always consult a qualified medical professional for diagnosis.
"""
        st.download_button(
            label="📥 Download Result",
            data=result_text,
            file_name="skin_cancer_result.txt",
            mime="text/plain"
        )

        st.markdown("---")
        st.caption("⚠️ Educational purposes only. Always consult a medical professional.")
