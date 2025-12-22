<div align="center">

<!-- Replace with your own banner/logo path -->
<img src="banner.svg" alt="SaliencySlider Logo">

*Interactively explore Grad-CAM saliency maps for VGG19 image classification.*

<!-- Badges (edit links as needed) -->
![Python](https://img.shields.io/badge/python-3.10-orange)
[![Model](https://img.shields.io/badge/model-VGG19-6A5ACD)](https://keras.io/api/applications/vgg/)
<a href="./SaliencySlider.pdf">
  <img alt="Final Report PDF" src="https://img.shields.io/badge/Final%20Report-PDF-red?logo=adobeacrobatreader&logoColor=white">
</a>
<a href="https://colab.research.google.com/drive/1xJEuaht0o6cHeA3eo6A3KWSu14sSp-wY?usp=sharing">
  <img alt="Google Colab" src="https://img.shields.io/badge/Try%20it-Google%20Colab-F9AB00?logo=googlecolab&logoColor=white">
</a>

</div>


### 🧠 Overview

**SaliencySlider** is a web application that lets you upload an image and **interactively explore** how a pretrained **VGG19** convolutional neural network makes classification decisions.  
A saliency slider controls how much of the **most influential image regions** (via Grad-CAM style saliency) are revealed, helping you understand what the model is “looking at.”


## ✨ Features

- **Image Upload**: Upload your own images for analysis.
- **Interactive Saliency Exploration**: Slider controls how many influential regions are visible.
- **Pretrained VGG19 CNN**: Uses a robust pretrained model for classification.


## Quick Start

You can experiment in Google Colab, or run locally via Docker.

## 🐳 Deployment (Docker)

Make sure Docker Desktop / Docker Daemon is running.

Build:

```bash
docker build -t saliency-slider-app .
```

Run:

```bash
docker run -d -p 8000:8000 saliency-slider-app
```


## 🧩 Tech Stack

| Component     | Technology |
| ------------- | ---------- |
| 🖥️ Web App    | Django     |
| 🧠 Model      | VGG19      |
| 🔍 Explainability | Grad-CAM / Saliency Maps |
| 🐳 Deployment | Docker     |
| 📓 Notebook   | Google Colab |


## 👥 Contributors

- Matheus Kunzler Maldaner — [GitHub](https://github.com/matheusmaldaner)  
- Kian Ambrose — [GitHub](https://github.com/kianambrose)  
- Lexie Certo  
- Kristian O'Connor — [GitHub](https://github.com/kroc99)  
