import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# PAGE CONFIG

st.set_page_config(page_title="DFU Detection + Grad-CAM", layout="wide")
st.title("Diabetic Foot Ulcer Detection with Grad-CAM")
st.write("Upload a foot image to get prediction and Grad-CAM visualization.")


# LOAD MODEL

MODEL_PATH = "mobilenetv2_binary_v3.h5"
last_conv_layer_name = "out_relu"

@st.cache_resource
def load_dfu_model():
    model = load_model(MODEL_PATH)
    return model

model = load_dfu_model()


# IMAGE PREPROCESSING

def preprocess_pil_image(pil_img, target_size=(224, 224)):
    img = pil_img.convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img).astype(np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array


# GRAD-CAM HEATMAP

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(img_array)
        class_channel = preds[:, 0]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    if max_val > 0:
        heatmap /= max_val

    return heatmap.numpy()


# OVERLAY HEATMAP

def overlay_gradcam_on_image(original_pil, heatmap, alpha=0.4):
    original_img = np.array(original_pil.convert("RGB"))

    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = Image.fromarray(np.uint8(jet_heatmap * 255))
    jet_heatmap = jet_heatmap.resize((original_img.shape[1], original_img.shape[0]))
    jet_heatmap = np.array(jet_heatmap)

    superimposed_img = np.uint8(alpha * jet_heatmap + original_img)
    return superimposed_img


# CONFIDENCE LABEL

def get_confidence_label(prob):
    distance = abs(prob - 0.5)

    if distance >= 0.4:
        return "High"
    elif distance >= 0.25:
        return "Medium"
    else:
        return "Low"


# FILE UPLOADER

uploaded_file = st.file_uploader("Choose a foot image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    pil_img = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded Image")
        st.image(pil_img, use_container_width=True)

    # preprocess and predict
    img_array = preprocess_pil_image(pil_img)
    pred_prob = float(model.predict(img_array, verbose=0)[0][0])
    pred_label = "Ulcer" if pred_prob > 0.5 else "Non-Ulcer"
    confidence = get_confidence_label(pred_prob)

    # grad-cam
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
    gradcam_img = overlay_gradcam_on_image(pil_img, heatmap, alpha=0.4)

    with col2:
        st.subheader("Prediction")
        st.write(f"**Class:** {pred_label}")
        st.write(f"**Probability:** {pred_prob:.4f}")
        st.write(f"**Confidence:** {confidence}")

        if confidence == "High":
            st.success("High confidence prediction")
        elif confidence == "Medium":
            st.warning("Medium confidence prediction")
        else:
            st.error("Low confidence prediction")

    st.subheader("Grad-CAM Visualization")
    st.image(gradcam_img, caption="Grad-CAM Overlay", use_container_width=True)

    # optional: show raw heatmap too
    st.subheader("Raw Heatmap")
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(heatmap, cmap="jet")
    ax.axis("off")
    st.pyplot(fig)