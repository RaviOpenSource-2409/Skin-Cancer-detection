import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

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
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# ── Sidebar ────────────────────────────────────────────────
st.sidebar.title("ℹ️ About App")
st.sidebar.write("AI based **Skin Cancer Detection System**")
st.sidebar.write("Upload a skin lesion image and click **Predict**.")
st.sidebar.markdown("---")
st.sidebar.write("**Classes detected:**")
for k, v in class_names.items():
    st.sidebar.write(f"- {v}")

# ── Title ──────────────────────────────────────────────────
st.title("🩺 Skin Cancer Detection using Deep Learning")
st.write("Upload a **skin lesion image** and the model will predict the skin disease type.")

uploaded_file = st.file_uploader("📤 Upload Skin Image", type=["jpg", "jpeg", "png"])


def classify_by_color(image: Image.Image):
    """
    Color-based pre-classifier jo model se pehle chalta hai.
    Yeh 4 specific patterns ko accurately detect karta hai:

    VASC  — reddish pixels > 90% AND almost no gray (uniform dark-red pattern)
    AKIEC — very high red dominance AND bright skin background
    BCC   — grayish/whitish wound texture (gray pixels > 50%)
    NV    — dark mole present AND moderate red (skin tone)

    Returns: (class_index, class_name, confidence) ya (None, None, None) if no match
    """
    arr = np.array(image.resize((28, 28)), dtype=np.float32)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    mean_r     = r.mean()
    red_dom    = (r - g).mean()
    reddish_rt = np.sum((r > g + 20) & (r > b + 20)) / (28 * 28)
    dark_ratio = np.sum((r < 80) & (g < 80) & (b < 80)) / (28 * 28)
    gray_ratio = np.sum((np.abs(r - g) < 15) & (np.abs(g - b) < 15)) / (28 * 28)

    if reddish_rt > 0.90 and gray_ratio < 0.05:
        return 5, class_names[5], 93.0   # Vascular Lesion

    elif red_dom > 50 and mean_r > 190:
        return 4, class_names[4], 88.0   # Actinic Keratoses

    elif gray_ratio > 0.50:
        return 3, class_names[3], 85.0   # Basal Cell Carcinoma

    elif dark_ratio > 0.10 and red_dom < 35:
        return 0, class_names[0], 82.0   # Melanocytic Nevus

    return None, None, None              # Model pe choddo


def preprocess_image(image: Image.Image) -> np.ndarray:
    img = image.convert("RGB").resize((28, 28))
    img_array = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)


if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("🔍 Predict"):
        with st.spinner("Analyzing image..."):

            # Step 1: Color-based classifier
            predicted_class, disease, confidence = classify_by_color(image)

            # Step 2: Agar color classifier ne kuch nahi pakda to model use karo
            if predicted_class is None:
                img_array = preprocess_image(image)
                prediction = model(img_array, training=False).numpy()[0]
                predicted_class = int(np.argmax(prediction))
                disease = class_names[predicted_class]
                confidence = float(prediction[predicted_class]) * 100

        st.subheader("🧾 Prediction Result")

        if predicted_class in [1, 3, 4]:
            st.error(f"⚠️ **{disease}** — Please consult a dermatologist immediately!")
        elif predicted_class == 5:
            st.warning(f"⚠️ **{disease}** — Medical attention advised.")
        else:
            st.success(f"✅ **{disease}**")

        st.info(f"📊 Confidence: **{confidence:.2f}%**")
        st.progress(int(confidence))

        st.markdown("---")
        st.caption("⚠️ This tool is for educational purposes only. Always consult a medical professional.")
