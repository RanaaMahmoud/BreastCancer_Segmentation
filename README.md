# 🧬 Breast Cancer Region Segmentation App

This is a Streamlit web app for automatically segmenting breast cancer regions in mammogram images using a U-Net deep learning model.
Upload a grayscale mammogram, and the model will highlight suspected cancerous regions.

## 🚀 Try It Out

👉 **Live Demo:** [Bone Fracture Detection Web App]([https://bonefracturedetection-app-app-odmvyyqvucylax43q9vafp.streamlit.app/](https://breastcancersegmentation-zjzr9mkmtawffblusmfhtp.streamlit.app/)

---

## 🔍 Features

* Upload a **grayscale** mammogram image (`.png`, `.jpg`, or `.jpeg`)
* Get:

  * The **original image**
  * The **predicted segmentation mask**
  * An **overlay** showing the cancerous region in **red**
* Powered by a pretrained **U-Net** model with PyTorch

---

## 📷 Example Output

| Original Image                | Overlayed Prediction         | Predicted Mask            |
| ----------------------------- | ---------------------------- | ------------------------- |
| ![](docs/sample-original.png) | ![](docs/sample-overlay.png) | ![](docs/sample-mask.png) |

---

## 🛠️ Setup Locally

### 🔧 1. Clone the Repo

```bash
git clone https://github.com/your-username/breast-cancer-segmentation.git
cd breast-cancer-segmentation
```

### 📦 2. Install Requirements

We recommend using a virtual environment.

```bash
pip install -r requirements.txt
```

### ⬇️ 3. Download Model Weights (Auto)

When you run the app, it will automatically download the model weights from Google Drive.

---

## 🚀 Run the App

```bash
streamlit run app.py
```

---

## 🧠 Model Details

* Architecture: [U-Net]
* Input Size: 128×128
* Trained on: Preprocessed grayscale mammograms
* Output: Segmentation mask (pixel-wise prediction)



