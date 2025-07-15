# Anemia_prediction

# Non-Invasive Anemia Detection using Deep Learning (DenseNet121, ResNet50, VGG16)

This project implements a **non-invasive, real-time anemia detection system** using computer vision and deep learning techniques. It leverages **DenseNet121**, **ResNet50**, **VGG16**, and **Naïve Bayes** to classify anemia from images of **conjunctiva**, **palm**, and **fingernails**. The project integrates **Mediapipe**, **Roboflow**, and **OpenCV** for segmentation and real-time inference.

---

## 📌 Highlights from the Research

> “This framework eliminates the need for blood testing, making anemia detection rapid, simple, and scalable in low-resource settings. It contributes to SDG Goal 3: Good Health and Well-Being.”  


---

## 🧠 Model Architectures

- **DenseNet121**: Core model used for real-time classification.
- **ResNet50 & VGG16**: Evaluated comparatively on conjunctiva dataset.
- **Naïve Bayes**: Used as a classical ML baseline on DenseNet-extracted features.

---

## 🔬 Technologies Used

| Category        | Tools / Frameworks                         |
|-----------------|--------------------------------------------|
| Deep Learning   | TensorFlow, Keras (DenseNet121, ResNet50, VGG16) |
| Preprocessing   | OpenCV, Roboflow, Mediapipe                |
| Backend         | Flask                                      |
| Frontend        | HTML, CSS, JavaScript                      |


---

## 🗂️ Dataset

- **Total Images**: 12,818
- **Modalities**: Palm, Nail, Conjunctiva
- **Classes**: Anemic, Non-Anemic
- **Image Size**: Resized to `128x128` and normalized to `[0, 1]`

---

## 🚀 Pipeline Overview

1. **Image Acquisition**  
   Real-time input via webcam or preloaded dataset.

2. **Segmentation**  
   - **Palm**: Extracted using Mediapipe hand landmarks.
   - **Nail & Eye**: Segmented using Roboflow-hosted YOLOv8.

3. **Classification**  
   - Deep learning models trained for binary classification using sigmoid activation.
   - Naïve Bayes classifier trained using DenseNet feature extractor output.

4. **Evaluation Metrics**  
   - Accuracy (Training & Validation)
   - Epoch-wise comparison plots
   - Grouped model performance chart

---



## 📊 Results

| Model       | Training Accuracy | Validation Accuracy |
|-------------|-------------------|---------------------|
| **Palm**    | 99.18%            | 96.95%              |
| Fingernail  | 97.30%            | 94.48%              |
| Conjunctiva | 97.39%            | 90.74%              |

---

## 🖥️ How to Run

```bash
# Clone the repo
git clone https://github.com/your-username/non-invasive-anemia-detection.git
cd non-invasive-anemia-detection

# Install dependencies
pip install -r requirements.txt

# Train & Evaluate
python anemia_detection.py

# Launch Web App (if implemented)
python app.py
```
# 📈 Visualizations
- ✅ Grouped line chart comparing model performance
<img width="450" height="205" alt="image" src="https://github.com/user-attachments/assets/04250def-9172-4025-b47b-f7fa5bbaa2c0" />
<img width="450" height="207" alt="image" src="https://github.com/user-attachments/assets/c0cac2bf-763f-439c-a6ba-93d92363c794" />
<img width="450" height="205" alt="image" src="https://github.com/user-attachments/assets/059b9aec-4abe-4d03-a7a4-e8ec27734b8e" />

- ✅ Epoch-wise training vs. validation accuracy plots
<img width="445" height="238" alt="image" src="https://github.com/user-attachments/assets/96b3599d-8e27-4548-aace-3e76d30b3da4" />


- ✅ Real-time classification via webcam feed (OpenCV + IP Webcam) **Palm**
  
<img width="248" height="135" alt="image" src="https://github.com/user-attachments/assets/05976ffc-af8e-4acb-a168-8c4c3f95da5f" />  <img width="196" height="143" alt="image" src="https://github.com/user-attachments/assets/13cb8571-cdd2-4cc2-a6a0-937f1f602d0e" />

<img width="438" height="107" alt="image" src="https://github.com/user-attachments/assets/d17902dc-c28b-47f7-9490-41d976890217" />

- ✅ Real-time classification by uploading photos via detected by roboflow API  **Conjunctiva**

<img width="211" height="122" alt="image" src="https://github.com/user-attachments/assets/bbb8c9cd-bc8e-45ab-8ffd-b697fa95a089" /> <img width="216" height="132" alt="image" src="https://github.com/user-attachments/assets/d8542824-e213-48a7-8bd0-04d430abd696" />

<img width="450" height="110" alt="image" src="https://github.com/user-attachments/assets/ae62456f-9f88-463f-a371-e77e65f9f316" />

- ✅ Real-time classification by uploading photos via detected by roboflow API  **Nail**

  <img width="235" height="99" alt="image" src="https://github.com/user-attachments/assets/e6048cca-195c-4007-ae7d-91b4ccd456dd" /> <img width="99" height="97" alt="image" src="https://github.com/user-attachments/assets/bb89ada3-fc2a-4d0b-a5ac-4a3fde98758c" /> <img width="101" height="94" alt="image" src="https://github.com/user-attachments/assets/d417f9e4-f706-47d6-80ec-a9ba94301de7" />

<img width="437" height="109" alt="image" src="https://github.com/user-attachments/assets/f463ebea-626e-47f4-a3f2-733973261812" />




# 📜 Research Paper
- Title: A DenseNet-CNN Framework for Non-Invasive Anemia Prediction Using Mediapipe and Roboflow-Based Feature Extraction
- Authors: K Vigneh, V Harish, M Abhilash, B Baranitharan, K Janahan, S Harish
- Affiliation: Kalasalingam Academy of Research and Education, India
- Status: Published

# 🤝 Contributions
If you’d like to improve segmentation, add additional biomarkers, or extend to other diseases, feel free to fork and contribute!

