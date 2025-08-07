# 👁️ Glaucoma Detection Using VGG16

## 📘 Overview

This project, **Glaucoma Detection Using VGG16**, was developed as part of the final semester **major project** in the B.Tech program (Artificial Intelligence and Machine Learning) at **Srinivas University Institute of Engineering and Technology**.

The aim is to aid in the **early detection of glaucoma**, a serious and often undiagnosed eye disease, by leveraging **deep learning** and **transfer learning**. The model uses the **VGG16 convolutional neural network** to classify **retinal fundus images** as **glaucomatous** or **non-glaucomatous**.

---

## 🧠 Problem Statement

Glaucoma is a silent thief of vision that damages the optic nerve irreversibly. Due to its gradual onset, early diagnosis is often missed. This project builds an **AI-assisted tool** to support ophthalmologists in **automated glaucoma screening** using retinal images.

---

## 🎯 Objectives

- 📌 Automate glaucoma detection using deep learning.
- 📌 Apply transfer learning via VGG16 for high performance on limited data.
- 📌 Improve early diagnosis and assist in clinical decision-making.
- 📌 Showcase deep learning applications in medical imaging.

---

## 📂 Dataset

- **Type**: Retinal Fundus Images  
- **Classes**: Glaucomatous, Non-Glaucomatous  
- **Format**: JPG/PNG  
- **Preprocessing**:
  - Resized to 224x224
  - Normalization
  - Augmentation (rotation, zoom, flipping)
  - Data split: Train / Validation / Test

> ⚠️ *Data sourced from anonymized public medical datasets.*

---

## 🧰 Technologies Used

- **Language**: Python  
- **Libraries**: TensorFlow, Keras, OpenCV  
- **Model**: VGG16 (Pre-trained on ImageNet)  
- **Tools**: Jupyter Notebook, Google Colab, PyCharm  

---

## 🏗️ Model Architecture

- Used **VGG16** as the base model (convolutional layers frozen)
- Added custom fully connected layers for binary classification
- Included **Dropout layers** to prevent overfitting
- Final layer with **sigmoid activation** for binary output

---

## ⚙️ Training & Evaluation

- **Optimizer**: Adam  
- **Loss Function**: Binary Cross-Entropy  
- **Metrics**: Accuracy, Precision, Recall, F1-Score  
- **Techniques**:
  - Data Augmentation
  - Early Stopping
  - Model Checkpointing

---

## 📈 Results

- **Test Accuracy**: ~90%  
- Evaluated using:
  - **Confusion Matrix**
  - **ROC-AUC Curve**
- Model accurately detects glaucoma in most test cases.

---

## ⚠️ Challenges Faced

1. **Limited and Imbalanced Dataset**
   - Medical datasets were small and had fewer glaucoma-positive samples.
   - Solution: Data augmentation and class weighting were applied.

2. **Overfitting**
   - The model initially overfit due to a small dataset.
   - Solution: Used Dropout, regularization, and early stopping.

3. **Image Quality Variance**
   - Fundus images varied in resolution and lighting.
   - Solution: Preprocessing with OpenCV and normalization helped standardize inputs.

4. **Model Tuning**
   - VGG16 has many layers; tuning without overfitting was challenging.
   - Solution: Froze base layers and experimented with custom dense layers.

5. **Understanding Medical Context**
   - Needed to understand glaucoma symptoms and visual cues.
   - Solution: Collaborated with medical professionals and studied medical papers.

---

## 🎤 How to Prepare for Interviews

When presenting this project in interviews, focus on:

- ✅ **Problem Explanation**:  
  “We aimed to detect glaucoma early using deep learning, as early symptoms are hard to spot clinically.”

- ✅ **Choice of VGG16**:  
  Discuss **transfer learning** and why VGG16 was ideal for working with small medical datasets.

- ✅ **Model Architecture**:  
  Be ready to sketch or explain how pre-trained layers were frozen and new layers were added for classification.

- ✅ **Evaluation Metrics**:  
  Know how and why **F1-score**, **precision**, and **ROC-AUC** were used, especially in medical imaging (where false negatives are critical).

- ✅ **Challenges Solved**:  
  Emphasize real-world challenges like data scarcity, overfitting, and image noise—and how you overcame them.

- ✅ **Ethical Awareness**:  
  Show awareness of ethical concerns like data privacy and clinical testing before deployment.

- ✅ **Future Plans**:  
  Mention plans like expanding to ResNet/EfficientNet or deploying the model via Flask/Streamlit.

---

## 🚀 Future Enhancements

- Integrate model into a **web application** for testing in clinical settings
- Expand dataset with more diverse demographics
- Explore **ResNet50**, **InceptionV3**, or **EfficientNet** for better results
- Collaborate with ophthalmologists for real-world validation

---

## ✅ Conclusion

This project demonstrates how **transfer learning** and **deep CNNs** can aid early detection of **glaucoma** through medical image analysis. With ~90% accuracy and careful model design, this AI tool can serve as a clinical decision support system, helping reduce undiagnosed cases of glaucoma.

---

## 👥 Contributors

- Joshmi K Joy  
- Aishwarya Ishwar Naik  
- Anusha Narayan Naik  
- G Raghavendra

---
