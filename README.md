# 🩺 VisionGuard AI

DEMO link : https://youtu.be/ijwzZ6_5Wzk

### Explainable AI for Early Diabetic Retinopathy Screening

VisionGuard AI is an AI-powered retinal screening prototype designed to assist in the early screening of diabetic retinopathy using retinal fundus images.

The system analyzes a retinal image, checks its quality, predicts the screening category using a CNN-based deep learning model, and provides an explainable visualization using Grad-CAM.

> 🚀 Smart India Hackathon • Health AI

---

## 🎯 Problem Statement

Diabetic retinopathy is a diabetes-related eye condition that can lead to vision loss if it is not detected early.

Traditional screening can require specialized equipment and trained professionals. VisionGuard AI aims to provide a simple digital screening workflow that can assist in identifying potential retinal abnormalities at an early stage.

---

## 💡 Our Solution

VisionGuard AI provides an end-to-end retinal screening workflow:

1. Enter patient information.
2. Upload a retinal fundus photograph.
3. Check the quality of the uploaded image.
4. Analyze the retinal image using a trained CNN model.
5. Display the predicted screening category and confidence.
6. Generate a Grad-CAM attention map for explainability.
7. Provide a patient-friendly explanation.
8. Store screening history using Firebase Firestore.
9. Generate a screening report.

---

## ✨ Key Features

- 🖼️ Retinal fundus image upload
- 🔍 Image quality gate
- 🧠 CNN-based diabetic retinopathy classification
- 📊 Model confidence score
- 🔥 Grad-CAM explainability
- 👤 Patient information management
- 🆔 Patient ID and Screening ID
- 📋 Screening history
- ☁️ Firebase Firestore integration
- 📄 Screening report generation
- 🌐 Web-based interface
- 🌍 Patient-friendly explanations in:
  - English
  - Kannada
  - Hindi

---

## 🧠 AI Pipeline

```text
Retinal Fundus Image
        ↓
Image Quality Check
        ↓
Image Preprocessing
        ↓
CNN / Deep Learning Model
        ↓
Prediction
        ↓
Confidence Score
        ↓
Grad-CAM Explainability
        ↓
Screening Result
        ↓
Patient-Friendly Explanation
