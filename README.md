# Image_Processing
# 📷 Photo & Video Processor

A powerful and user-friendly web application built with Streamlit that allows you to **capture**, **process**, and **record** photos and videos using your **laptop webcam** or **smartphone IP camera**. It also supports , custom image filters, and processed image saving features.

---

## 🚀 Features

- 🎥 **Live Camera Feed**
  - Use laptop/USB webcam or smartphone IP webcam.
  - Toggle between **normal mode** and **face tracking mode**.
  - Flip video for a selfie-friendly experience.

- 📸 **Capture Frame**
  - Capture high-quality frames from the live feed.
  - Automatically saves images for processing.

- 🧠 **Face Detection & Tracking**
  - Real-time face detection using OpenCV Haar cascades.
  - Face tracking using CSRT tracker for smooth tracking in video feed.

- 🖼️ **Image Processing Tools**
  - Crop, grayscale, blur, brightness & contrast adjustment.
  - Apply edge detection to get an image with eadges.

- 🎞️ **Video Recording**
  - Start and stop recording live video feed.
  - Automatically save and store videos in AVI format.

- 🧠 **Smart Switching**
  - Seamless switch between live camera and image processing view.

---

## 🛠️ Installation

```bash
git clone https://github.com/yourusername/photo-video-processor.git
cd photo-video-processor
pip install -r requirements.txt
streamlit run app.py
