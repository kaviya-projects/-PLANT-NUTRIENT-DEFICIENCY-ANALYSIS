# 🌾 Nutrient Deficiency Symptoms in Rice - Image Classification using CNN

This project uses deep learning (Convolutional Neural Networks) to identify nutrient deficiency in rice plants based on leaf images. The model classifies whether the rice plant is **deficient in Nitrogen (N)**, **Phosphorus (P)**, **Potassium (K)**, or **Healthy**.

---

## 📂 About the Dataset

- **Dataset Type:** Image classification
- **Classes:**
  - `Nitrogen_Deficiency` - 440 images
  - `Phosphorus_Deficiency` - 333 images
  - `Potassium_Deficiency` - 383 images
- **Source:** Leaf images showing visible signs of nutrient deficiency.
- **Format:** Directory-based image folders per class

> Each image depicts rice plants lacking one of the essential nutrients. No license info is specified.

---

## 🧠 Model

- **Framework:** TensorFlow (Keras)
- **Model File:** `plant_nutrient_model.h5`
- **Input Shape:** `(150, 150, 3)`
- **Architecture:** CNN-based custom model (trained separately)
- **Output:** 4-class softmax classifier

---

## 🔍 How It Works

### ✅ Class Labels

```python
class_labels = ['Nitrogen_Deficiency', 'Phosphorus_Deficiency', 'Potassium_Deficiency', 'Healthy']
```
## Output: 
<img width="1919" height="679" alt="Screenshot 2025-07-17 190623" src="https://github.com/user-attachments/assets/56a0aa82-0323-4082-b06f-b14983f02414" />
<img width="1919" height="1022" alt="Screenshot 2025-07-17 190657" src="https://github.com/user-attachments/assets/562a1a93-60f7-431d-8a6f-56608d451297" />
<img width="1917" height="983" alt="image" src="https://github.com/user-attachments/assets/53031f8c-a894-481b-b97d-4dbcb350f1bb" />

## ✅ Requirements

Install the required Python packages:

```bash
pip install tensorflow matplotlib numpy
```

## 📌 Tags

`Rice Plant`, `Image Classification`, `Nutrient Deficiency`, `Nitrogen`, `Phosphorus`, `Potassium`, `Deep Learning`, `Agriculture`, `TensorFlow`, `CNN`
