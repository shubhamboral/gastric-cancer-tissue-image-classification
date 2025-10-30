
# Gastric Cancer Histopathology Image Classification

This project builds a **Convolutional Neural Network (CNN)** using TensorFlow and Keras to classify histopathological tissue images into different gastric tissue types. The trained model is deployed using **Streamlit** on **Hugging Face Spaces** for interactive prediction.

---

## 📂 Dataset

**Name:** Gastric Cancer Histopathology Tissue Image Dataset  
**Source:** [Kaggle Dataset](https://www.kaggle.com/datasets/andrewmvd/gastric-cancer-histopathology-tissue-image-dataset)

### Folder Structure (after extraction):
```
gastric_data/
├── HMU-GC-HE-30K/
│   ├── all_image/
│   │   ├── ADI/
│   │   ├── DEB/
│   │   ├── LYM/
│   │   ├── MUC/
│   │   ├── MUS/
│   │   ├── NOR/
│   │   ├── STR/
│   │   └── TUM/
```

---

## 📌 Project Goals

- Classify different gastric tissue types from histopathology images
- Handle multiple classes (8 classes)
- Build and train a CNN model with early stopping and checkpointing
- Deploy for real-time inference using Streamlit on Hugging Face

---

## 🧪 Steps Performed

### 1. **Dataset Exploration & Visualization**
- Examined the folder and class distribution.
- Displayed sample images from each tissue class.
- Checked for class imbalance.

### 2. **Data Preprocessing**
- Resized images to **224x224 pixels**.
- Normalized pixel values to `[0, 1]`.
- Created training and validation datasets using `image_dataset_from_directory()`.

### 3. **Model Architecture**
```python
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dense(8, activation='softmax')  # 8-class classification
])
```

### 4. **Model Compilation**
```python
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```

### 5. **Training Setup**
- Trained for **up to 20 epochs** with **EarlyStopping**.
- Used **ModelCheckpoint** to save the best model (`bestmodel.keras`).

### 6. **Performance Evaluation**
- Calculated accuracy, precision, recall, F1-score.
- Generated a confusion matrix.
- Identified and analyzed misclassified images.

---

## 📊 Results

| Metric               | Value (Approx.) |
|----------------------|-----------------|
| Training Accuracy     | ↑ steadily up to ~0.85 |
| Validation Accuracy   | Stabilized ~0.70–0.75 |
| Validation Loss       | Controlled well with early stopping |

> The model showed good generalization. Some confusion between certain tissue types remains, indicating scope for further fine-tuning.

---

## 🧠 Key Learnings

- CNNs can effectively distinguish different tissue types in medical images.
- EarlyStopping and ModelCheckpoint improve model robustness.
- Deploying on Hugging Face makes the project easily accessible.

---

## 🖥️ Deployment

The trained model is deployed using **Streamlit** on **Hugging Face Spaces**.

🔗 **Access the deployed app here:**  
👉 [Gastric Cancer Detection App](https://huggingface.co/spaces/Usurper0/gastric_cancer_detection)

### Deployment Steps:
1. Trained model saved as `bestmodel.keras`.
2. Streamlit app (`app.py`) created to upload and predict images.
3. `requirements.txt` included for dependency management.
4. Uploaded all files (`app.py`, `bestmodel.keras`, `requirements.txt`) to Hugging Face Spaces.
5. App runs instantly in the browser!

---

## 🧰 Dependencies

- Python 3.8+
- TensorFlow
- Streamlit
- NumPy
- Pillow
- scikit-learn

---
