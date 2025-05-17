# Emotion Detection from Text

This project was developed as part of the ASE Trainee (AIML) selection process for Cyfuture India (Batch 2025).

## 📌 Objective
To build a machine learning model that can detect emotions such as joy, sadness, anger, fear, love, and surprise from text data using Python and AI/ML libraries.

---

## 🗃️ Dataset
- **Name**: Emotions Dataset for NLP  
- **Source**: [Kaggle - praveengovi/emotions-dataset-for-nlp](https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp)  
- **Format**: Semicolon-separated `.txt` files with `text;emotion`  
- **Classes**: joy, sadness, anger, fear, love, surprise

---

## 🛠️ Tools and Libraries
- Python
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- neattext
- joblib

---

## 🧠 ML Workflow
1. **Load dataset**
2. **Text preprocessing** using NeatText (stopwords, punctuation, user handles removal)
3. **TF-IDF Vectorization**
4. **Model training** using Logistic Regression
5. **Evaluation** using accuracy and classification report
6. **Prediction function** for real-time emotion detection
7. **Visualization** of emotion distribution

---

## ✅ Results
- Achieved **accuracy > 95%** on test data
- Accurate multi-class classification on emotions

---

## 📦 Files Included
- `emotion-detection-from-text.ipynb` – Main code notebook
- `emotion_model.pkl` – Trained model
- `vectorizer.pkl` – Saved TF-IDF vectorizer

---

## 🖥️ Run Instructions
1. Open the Jupyter Notebook or run in [Kaggle](https://www.kaggle.com/)
2. Upload the `train.txt` dataset file
3. Run all cells to train the model and make predictions

---
