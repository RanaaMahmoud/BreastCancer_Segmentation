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
        x = self.conv2d_1(x)
        x = self.batchnorm_1(x)
        x = self.relu_1(x)
        
        x = self.dropout(x)
        
        x = self.conv2d_2(x)
        x = self.batchnorm_2(x)
        x = self.relu_2(x)

        return x

class Encoder(nn.Module):
    def __init__(self, input_channel, out_channel, dropout):
        super(Encoder, self).__init__()
        self.conv2d_1 = ConvBlock(input_channel, out_channel, dropout)
        self.maxpool = nn.MaxPool2d((2,2))
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.conv2d_1(x)
        p = self.maxpool(x)
        p = self.dropout(p)

        return x, p

class Decoder(nn.Module):
    def __init__(self, input_channel, output_channel, dropout):
        super(Decoder, self).__init__()
        self.conv_t = nn.ConvTranspose2d(input_channel, output_channel, stride=2, kernel_size=2)
        self.conv2d_1 = ConvBlock(output_channel*2, output_channel, dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, skip):
        x = self.conv_t(x)
        x = torch.cat([x, skip], dim=1)
        x = self.dropout(x)
        x = self.conv2d_1(x)

        return x

# ---------------- Define Full UNet ----------------
class Unet(nn.Module):

    def __init__(self, input_channel=1):
        super().__init__()
        self.encoder_1 = Encoder(input_channel, 64, 0.07)
        self.encoder_2 = Encoder(64, 128, 0.08)
        self.encoder_3 = Encoder(128, 256, 0.09)
        self.encoder_4 = Encoder(256, 512, 0.1)

        self.conv_block = ConvBlock(512, 1024, 0.11)

        self.decoder_1 = Decoder(1024, 512, 0.1)
        self.decoder_2 = Decoder(512, 256, 0.09)
        self.decoder_3 = Decoder(256, 128, 0.08)
        self.decoder_4 = Decoder(128, 64, 0.07)

        self.cls = nn.Conv2d(64, 1, kernel_size=1, padding=0)
        self.relu = nn.Sigmoid() 

    def forward(self, x):

        """ ------ Encoder ------"""
        x1, p1 = self.encoder_1(x)
        x2, p2 = self.encoder_2(p1)
        x3, p3 = self.encoder_3(p2)
        x4, p4 = self.encoder_4(p3)

        """ ------ BottleNeck ------"""
        x5 = self.conv_block(p4)

        """ ------ Decoder ------"""
        x6 = self.decoder_1(x5, x4)
        x7 = self.decoder_2(x6, x3)
        x8 = self.decoder_3(x7, x2)
        x9 = self.decoder_4(x8, x1)
        
        """ ------ Final Layer ------"""
        x_final = self.cls(x9)
        x_final = self.relu(x_final)

        return x_final

# ---------------- Download Model if Needed ----------------
MODEL_PATH = "model_weights.pth"
GDRIVE_FILE_ID = "1GHcCccSW7v7wxbrWZ8uE07BwtcR9pUNR"  # <- Replace with your actual ID

if not os.path.exists(MODEL_PATH):
    with st.spinner("🔽 Downloading model weights..."):
        gdown.download(f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}", MODEL_PATH, quiet=False, use_cookies=False)


# ---------------- Load Model ---------------
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
#Define image transformation (same as your val_transforms)
image_size = 128
val_transforms = transforms.Compose([
    transforms.Resize([image_size, image_size]),
    transforms.ToTensor()
])

# Overlay mask on original image
def overlay_mask(image: Image.Image, mask_tensor: torch.Tensor) -> Image.Image:
    # Resize original image
    image = image.resize((image_size, image_size))

    # Convert tensor to numpy
    mask = mask_tensor.squeeze().cpu().numpy()
    mask = np.clip(mask * 255, 0, 255).astype(np.uint8)

    # Convert to grayscale PIL image (important!)
    mask_img = Image.fromarray(mask).convert("L")

    # Skip colorizing if the mask is blank
    if mask_img.getextrema()[1] == 0:
        return image.convert("RGB")

    # Colorize and blend
    st.write("Mask image extrema (min/max):", mask_img.getextrema())
    color_mask = ImageOps.colorize(mask_img, black="black", white="red").convert("RGBA")
    base_img = image.convert("RGBA")
    blended = Image.blend(base_img, color_mask, alpha=0.4)
    return blended



# Streamlit UI
st.title("Segmentation Inference Demo")
uploaded_file = st.file_uploader("Upload a grayscale image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')
    st.subheader("Original Image")
    st.image(image, width=256)

    # Preprocess
    input_tensor = val_transforms(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(input_tensor)
        # Optional: Apply sigmoid if your model output is not already thresholded
        pred_mask = torch.sigmoid(output)
        pred_mask = (pred_mask > 0.5).float()  # Binarize

    # Overlay
    result = overlay_mask(image, pred_mask[0])  # Remove batch

    st.subheader("Predicted Mask Overlay")
    st.image(result, width=256)