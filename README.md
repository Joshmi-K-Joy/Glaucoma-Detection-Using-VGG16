# Glaucoma Detection Using VGG16

## Overview

This project, *Glaucoma Detection Using VGG16*, was developed as part of our final semester major project in the BTech program (Artificial Intelligence and Machine Learning) at Srinivas University Institute of Engineering and Technology.

The primary goal of the project is to assist in the **early detection of glaucoma**, a serious eye disease that can lead to irreversible blindness, through **automated image classification** using deep learning. By leveraging the power of transfer learning with the **VGG16 convolutional neural network**, the model aims to classify retinal fundus images as glaucomatous or non-glaucomatous with improved accuracy and efficiency.

## Problem Statement

Glaucoma is a silent eye disease that causes progressive and irreversible damage to the optic nerve. Early diagnosis is crucial but often missed due to the subtle onset of symptoms. This project aims to support ophthalmologists and eye-care professionals by providing an AI-based system that can aid in the early detection of glaucoma through image analysis.


## Objectives

- To automate the detection of glaucoma from retinal fundus images.
- To utilize transfer learning with VGG16 for better performance with limited medical datasets.
- To improve diagnostic accuracy and support early intervention.
- To explore the application of deep learning in medical image analysis.


## Dataset

- **Type**: Retinal fundus images
- **Format**: JPEG/PNG
- **Classes**: Glaucomatous, Non-Glaucomatous
- **Preprocessing**:
  - Resizing images to 224x224 pixels
  - Normalization and augmentation
  - Splitting into training, validation, and test sets

> *Note: The dataset used is anonymized and sourced from publicly available medical datasets.*


## Technologies Used

- **Programming Language**: Python
- **Deep Learning Framework**: TensorFlow & Keras
- **Model Architecture**: VGG16 (Pre-trained on ImageNet)
- **Image Processing**: OpenCV
- **Development Tools**: Jupyter Notebook, Google Colab / PyCharm


## Model Architecture

- Pre-trained VGG16 used with frozen convolutional base
- Custom dense layers added for binary classification
- Dropout regularization to prevent overfitting
- Binary cross-entropy loss function
- Evaluation using accuracy, precision, recall, and F1-score


## Training and Evaluation

- Optimizer: Adam
- Loss Function: Binary Cross-Entropy
- Metrics: Accuracy, Precision, Recall, F1-Score
- Techniques used:
  - Early stopping
  - Model checkpointing
  - Data augmentation
- Achieved consistent performance in validation and test phases


## Results

- **Test Accuracy**: ~90% (depending on dataset size and preprocessing)
- **Confusion Matrix** and **ROC-AUC** were used to evaluate performance
- The model successfully detected early signs of glaucoma in fundus images


## Future Enhancements

- Integration into a real-time web application for clinical testing
- Expand dataset for greater diversity and generalization
- Experiment with other architectures (e.g., ResNet50, EfficientNet)
- Collaborate with medical institutions for feedback and validation


## Conclusion

This project demonstrates the potential of deep learning, especially transfer learning, in supporting early disease detection in healthcare. By automating glaucoma detection, we aim to provide an assistive tool for medical professionals and contribute to preventive eye care.


## Contributors

- Joshmi K Joy  
- Aishwarya Ishwar Naik
- Anusha Narayan Naik
- G Raghavendra




 
