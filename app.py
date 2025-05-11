import streamlit as st
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from datetime import datetime
import os
import gdown

# ---------------- Streamlit Page Config ----------------
st.set_page_config(page_title="🩻 Breast Segmentation", layout="centered")
st.title("🧬 Breast Cancer Region Segmentation")
st.markdown("Upload a mammogram image to visualize the segmented cancerous region.")

# ---------------- Download Model if Needed ----------------
MODEL_PATH = "model.pt"
GDRIVE_FILE_ID = "YOUR_FILE_ID_HERE"  # <- Replace with your real file ID

if not os.path.exists(MODEL_PATH):
    with st.spinner("🔽 Downloading model weights..."):
        gdown.download(f"https://drive.google.com/uc?id={14HEwfJQjVdn7VIyRFu-byjXKkTYiMbpJ}", MODEL_PATH, quiet=False)

# ---------------- Load Model ----------------
@st.cache_resource
def load_model():
    model = torch.load(MODEL_PATH, map_location=torch.device('cpu'))  # Use GPU if available
    model.eval()
    return model

model = load_model()

# ---------------- Preprocessing ----------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# ---------------- Segmentation Function ----------------
def segment_image(image: Image.Image, model):
    img_tensor = transform(image).unsqueeze(0)  # shape: [1, 1, 128, 128]
    if img_tensor.shape[1] == 3:
        img_tensor = img_tensor[:, 0:1]  # Ensure single channel if accidentally RGB

    with torch.no_grad():
        output = model(img_tensor)
        if output.shape[1] == 1:
            output = torch.sigmoid(output)
            mask = (output > 0.5).float()
        else:
            mask = torch.argmax(output, dim=1).float()

    return mask.squeeze(0).squeeze().numpy()

# ---------------- Upload + Display ----------------
uploaded_file = st.file_uploader("📤 Upload a grayscale mammogram image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    pil_image = Image.open(uploaded_file).convert("L")  # Ensure grayscale
    st.image(pil_image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("🧠 Running segmentation..."):
        mask_np = segment_image(pil_image, model)

    # Display segmentation mask
    st.image(mask_np, caption="🩻 Segmented Region", use_column_width=True, clamp=True)

    # Save option
    if st.checkbox("💾 Save segmentation mask as image"):
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        mask_img = Image.fromarray((mask_np * 255).astype(np.uint8))
        filename = f"mask_{timestamp}.png"
        mask_img.save(filename)
        st.success(f"✅ Mask saved as {filename}")

else:
    st.info("👆 Please upload a mammogram image to begin.")
