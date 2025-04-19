
# Gastric Cancer Histopathology Image Classification

This project involves building a **Convolutional Neural Network (CNN)** using TensorFlow and Keras to classify histopathological tissue images into cancerous or non-cancerous categories. It is trained on a public dataset of gastric cancer tissue images and designed to be deployed using Streamlit for interactive inference.

---

## 📂 Dataset

**Name:** Gastric Cancer Histopathology Tissue Image Dataset  
**Source:** [Kaggle Dataset](https://www.kaggle.com/datasets/andrewmvd/gastric-cancer-histopathology-tissue-image-dataset)

### Folder Structure (after extraction):
```
gastric_data/
├── train/
│   ├── cancer/
│   └── non-cancer/
├── test/
│   ├── cancer/
│   └── non-cancer/
└── valid/
    ├── cancer/
    └── non-cancer/
```

---

## 📌 Project Goals

- Classify gastric cancer vs non-cancerous images
- Understand and handle class imbalance
- Train a CNN model with early stopping
- Evaluate and visualize performance
- Prepare for Streamlit deployment

---

## 🧪 Steps Performed

### 1. **Dataset Exploration & Visualization**
- Examined directory structure.
- Displayed class-wise image samples.
- Identified class imbalance.
- Visualized some histopathological images.

### 2. **Data Preprocessing**
- Resized all images to 150x150 pixels.
- Normalized pixel values to `[0, 1]`.
- Created `train_ds`, `val_ds`, and `test_ds` using `image_dataset_from_directory()`.

### 3. **Model Architecture**
```python
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification
])
```

### 4. **Model Compilation**
```python
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
```

### 5. **Training Setup**
- Trained for **20 epochs** (or until early stopping).
- Added `EarlyStopping` and `ModelCheckpoint`.
- Monitored both training and validation metrics.

### 6. **Performance Evaluation**
- Calculated: accuracy, precision, recall, F1-score.
- Generated confusion matrix.
- Identified and visualized misclassified images.

---

## 📊 Results

| Metric               | Value (Approx.) |
|----------------------|-----------------|
| Training Accuracy     | ↑ steadily up to ~0.70 |
| Validation Accuracy   | Stabilized ~0.55–0.57 |
| Validation Loss       | Showed slight overfitting trend |

> The training accuracy increased with each epoch while validation accuracy plateaued, suggesting room for model or data augmentation improvements.

---

## 🧠 Key Learnings

- **CNNs are effective** for visual medical diagnosis tasks.
- **Early stopping** prevents overfitting on noisy medical data.
- Model generalization remains a challenge with limited data.
- Dataset may require augmentation or more samples for higher accuracy.

---

## 🖥️ Deployment

You can deploy the trained model using [Streamlit](https://streamlit.io/):
1. Save the trained model:
   ```python
   model.save("gastric_cancer_model.h5")
   ```
2. Create `app.py` in Streamlit and load the model:
   ```python
   model = load_model('gastric_cancer_model.h5')
   ```
3. Use `st.file_uploader` and `PIL` to take input and predict.

---

## 🧰 Dependencies

- Python 3.8+
- TensorFlow
- Matplotlib
- NumPy
- scikit-learn
- Streamlit (for deployment)
- OpenCV or PIL (for image handling)

---

## ⚠️ Ethical Considerations

- Medical AI tools **must not replace** professional diagnosis.
- Results must be validated clinically before real-world use.
- Transparency and accountability are essential in healthcare ML systems.

---

## 📃 License

This project is for educational and research purposes only.

---

## ✍️ Author

**Shubham Boral**  
CSE Pre-Final Year Student | Interested in AI + Medical Imaging + Cybersecurity  

---
