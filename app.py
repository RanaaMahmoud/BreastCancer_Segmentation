import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from PIL import Image
from datetime import datetime
import os
import gdown
import matplotlib.pyplot as plt

# ---------------- Streamlit Config ----------------
st.set_page_config(page_title="🩻 Breast Segmentation", layout="centered")
st.title("🧬 Breast Cancer Region Segmentation")
st.markdown("Upload a mammogram image to visualize the segmented cancerous region.")

# ---------------- Define UNet Blocks ----------------
class ConvBlock(nn.Module):
    def __init__(self, input_channel, out_channel, dropout):
        super(ConvBlock, self).__init__()
        self.conv2d_1 = nn.Conv2d(input_channel, out_channel, kernel_size=3, padding=1)
        self.batchnorm_1 = nn.BatchNorm2d(out_channel)
        self.relu_1 = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2d_2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1)
        self.batchnorm_2 = nn.BatchNorm2d(out_channel)
        self.relu_2 = nn.ReLU()

    def forward(self, x):
        x = self.relu_1(self.batchnorm_1(self.conv2d_1(x)))
        x = self.dropout(x)
        x = self.relu_2(self.batchnorm_2(self.conv2d_2(x)))
        return x

class Encoder(nn.Module):
    def __init__(self, input_channel, out_channel, dropout):
        super(Encoder, self).__init__()
        self.conv = ConvBlock(input_channel, out_channel, dropout)
        self.pool = nn.MaxPool2d((2,2))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv(x)
        p = self.dropout(self.pool(x))
        return x, p

class Decoder(nn.Module):
    def __init__(self, input_channel, output_channel, dropout):
        super(Decoder, self).__init__()
        self.conv_t = nn.ConvTranspose2d(input_channel, output_channel, stride=2, kernel_size=2)
        self.conv = ConvBlock(output_channel*2, output_channel, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, skip):
        x = self.conv_t(x)
        x = torch.cat([x, skip], dim=1)
        x = self.dropout(x)
        x = self.conv(x)
        return x

# ---------------- Define Full UNet ----------------
class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        self.encoder1 = Encoder(1, 64, 0.1)
        self.encoder2 = Encoder(64, 128, 0.1)
        self.encoder3 = Encoder(128, 256, 0.2)
        self.encoder4 = Encoder(256, 512, 0.2)
        self.bottleneck = ConvBlock(512, 1024, 0.3)
        self.decoder1 = Decoder(1024, 512, 0.2)
        self.decoder2 = Decoder(512, 256, 0.2)
        self.decoder3 = Decoder(256, 128, 0.1)
        self.decoder4 = Decoder(128, 64, 0.1)
        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        s1, p1 = self.encoder1(x)
        s2, p2 = self.encoder2(p1)
        s3, p3 = self.encoder3(p2)
        s4, p4 = self.encoder4(p3)
        b = self.bottleneck(p4)
        d1 = self.decoder1(b, s4)
        d2 = self.decoder2(d1, s3)
        d3 = self.decoder3(d2, s2)
        d4 = self.decoder4(d3, s1)
        return self.final(d4)

# ---------------- Download Model if Needed ----------------
MODEL_PATH = "model_weights.pth"
GDRIVE_FILE_ID = "1GHcCccSW7v7wxbrWZ8uE07BwtcR9pUNR"  # <- Replace with your actual ID

if not os.path.exists(MODEL_PATH):
    with st.spinner("🔽 Downloading model weights..."):
        gdown.download(f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}", MODEL_PATH, quiet=False, use_cookies=False)

# ---------------- Load Model ---------------
# ---------------- Load Model ----------------
@st.cache_resource
def load_model():
    try:
        # Rebuild the architecture
        model = Unet(input_channel=1)
        state_dict = torch.load(MODEL_PATH, map_location="cpu")  # Now this is weights only
        model.load_state_dict(state_dict)
        model.eval()
        st.success("✅ Model weights loaded and architecture rebuilt!")
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None

model = load_model()
# ---------------- Preprocessing ----------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# ---------------- Inference ----------------
def segment_image(image: Image.Image, model):
    img_tensor = transform(image).unsqueeze(0)
    if img_tensor.shape[1] == 3:
        img_tensor = img_tensor[:, 0:1]  # convert RGB to single channel if needed

    with torch.no_grad():
        output = model(img_tensor)
        output = torch.sigmoid(output)
        mask = (output > 0.5).float()
    return mask.squeeze().numpy()

# ---------------- Display Mask ----------------
def display_mask(mask_np):
    fig, ax = plt.subplots()
    ax.imshow(mask_np, cmap='hot')
    ax.axis('off')
    st.pyplot(fig)

# ---------------- Upload + Display ----------------
uploaded_file = st.file_uploader("📤 Upload a grayscale mammogram image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    pil_image = Image.open(uploaded_file).convert("L")  # force grayscale
    st.image(pil_image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("🧠 Running segmentation..."):
        mask_np = segment_image(pil_image, model)

    display_mask(mask_np)

    if st.checkbox("💾 Save segmentation mask as image"):
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        mask_img = Image.fromarray((mask_np * 255).astype(np.uint8))
        filename = f"mask_{timestamp}.png"
        mask_img.save(filename)
        st.success(f"✅ Mask saved as {filename}")
else:
    st.info("👆 Please upload a mammogram image to begin.")
