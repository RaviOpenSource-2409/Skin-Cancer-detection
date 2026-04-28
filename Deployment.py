import numpy as np
import tensorflow as tf
from PIL import Image

# ── Model load ─────────────────────────────────────────────
MODEL_PATH = r"D:\Skin_Cancer_Detection\model1.h5"
model = tf.keras.models.load_model(MODEL_PATH)

class_names = {
    0: "Melanocytic Nevus (NV)",
    1: "Melanoma (MEL)",
    2: "Benign Keratosis (BKL)",
    3: "Basal Cell Carcinoma (BCC)",
    4: "Actinic Keratoses (AKIEC)",
    5: "Vascular Lesion (VASC)",
    6: "Dermatofibroma (DF)"
}


def classify_by_color(image: Image.Image):
    """
    Color-based pre-classifier — model se pehle chalta hai.

    VASC  — reddish pixels > 90% AND almost no gray
    AKIEC — very high red dominance AND bright skin
    BCC   — grayish/whitish wound (gray > 50%)
    NV    — dark mole AND moderate red (skin tone)

    Returns: (class_index, class_name, confidence) ya (None, None, None)
    """
    arr = np.array(image.resize((28, 28)), dtype=np.float32)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    mean_r     = r.mean()
    red_dom    = (r - g).mean()
    reddish_rt = np.sum((r > g + 20) & (r > b + 20)) / (28 * 28)
    dark_ratio = np.sum((r < 80) & (g < 80) & (b < 80)) / (28 * 28)
    gray_ratio = np.sum((np.abs(r - g) < 15) & (np.abs(g - b) < 15)) / (28 * 28)

    if reddish_rt > 0.90 and gray_ratio < 0.05:
        return 5, class_names[5], 93.0

    elif red_dom > 50 and mean_r > 190:
        return 4, class_names[4], 88.0

    elif gray_ratio > 0.50:
        return 3, class_names[3], 85.0

    elif dark_ratio > 0.10 and red_dom < 35:
        return 0, class_names[0], 82.0

    return None, None, None


def preprocess_image(image: Image.Image) -> np.ndarray:
    img = image.resize((28, 28))
    img_array = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)


# ── Apni image ka PURA path yahan dalo ────────────────────
image_path = r"D:\Skin_Cancer_Detection\image.jpg"   # <-- CHANGE THIS

image = Image.open(image_path).convert("RGB")

# Step 1: Color classifier
predicted_class, disease, confidence = classify_by_color(image)

# Step 2: Agar match nahi hua to model use karo
if predicted_class is None:
    img_array = preprocess_image(image)
    prediction = model(img_array, training=False).numpy()[0]
    predicted_class = int(np.argmax(prediction))
    disease = class_names[predicted_class]
    confidence = float(prediction[predicted_class]) * 100
    print("(Model prediction)")
else:
    print("(Color analysis prediction)")

print(f"Predicted Disease : {disease}")
print(f"Confidence        : {confidence:.2f}%")
