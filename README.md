````md
# Sentiment Analysis Web App (Naive Bayes vs BERT)

This project is a full-stack AIML web application that performs sentiment analysis on text input using two different approaches:

1. Traditional Machine Learning – Naive Bayes with TF-IDF  
2. Deep Learning (NLP) – Fine-tuned BERT Transformer  

The application allows users to compare predictions, confidence scores, and overall model accuracy through a Flask-based web interface.

---

## Project Overview

Sentiment analysis is a Natural Language Processing (NLP) task used to determine whether a given text expresses a positive or negative opinion.

In this project:
- A baseline machine learning model (Naive Bayes) is implemented to demonstrate traditional text classification.
- An advanced transformer-based model (BERT) is used to show how deep learning models understand context and semantics.
- Both models are integrated into a single web application to allow direct comparison of their performance.

---

## Key Features

- Dual model support:
  - Naive Bayes (TF-IDF based)
  - BERT (Transformer-based deep learning model)
- Model selection option (Naive Bayes or BERT)
- Prediction confidence percentage
- Confidence visualization bar
- Side-by-side accuracy comparison of models
- Flask-based backend with HTML and CSS frontend
- BERT model trained using GPU on Google Colab
- Dataset used: IMDB Movie Reviews

---

## Technology Stack

### Backend and Machine Learning
- Python  
- Flask  
- Scikit-learn  
- NLTK  
- PyTorch  
- Hugging Face Transformers  

### Frontend
- HTML  
- CSS  

### Tools
- Git and GitHub  
- Google Colab (for BERT training)  

---

## Model Details

### Naive Bayes Model
- Feature extraction using TF-IDF  
- Classifier: Multinomial Naive Bayes  
- Fast and lightweight  
- Works on word frequency probabilities  
- Does not understand contextual relationships  

Accuracy: **86%**

---

### BERT Model
- Model: bert-base-uncased  
- Fine-tuned on the IMDB movie reviews dataset  
- Captures semantic meaning and contextual information  
- Performs better on complex and ambiguous sentences  

Accuracy: **91.5%**

---

## Model Performance Comparison

| Model        | Approach               | Accuracy |
|-------------|------------------------|----------|
| Naive Bayes | TF-IDF + MultinomialNB | 86%      |
| BERT        | Transformer-based NLP  | 91.5%    |

---

## Application UI Explanation

The web interface allows users to:
1. Enter text for sentiment analysis  
2. Select the model (Naive Bayes or BERT)  
3. View the predicted sentiment (Positive or Negative)  
4. See the confidence score for the prediction  
5. Observe a confidence bar representing model certainty  
6. Compare overall accuracy of both models  

The confidence score represents how sure the model is about a specific input, while accuracy represents overall performance on the dataset.

---

## How to Run the Project Locally

### Step 1: Clone the Repository
```bash
git clone https://github.com/saisri267/Sentiment-Analysis-Web-App.git
cd Sentiment-Analysis-Web-App
````

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Run the Application

```bash
python app.py
```

Open the application in a browser:

```
http://127.0.0.1:5000
```

---

## Note on Large Files

Due to GitHub file size limitations, large files are excluded from the repository:

* Trained BERT model weights
* IMDB dataset CSV file

To use the BERT model:

* Train the model using `train_bert.py` on Google Colab with GPU support
  OR
* Download the trained model separately and place it inside the `bert_model/` directory.

---

## Project Flow

1. Preprocess text data from the IMDB movie reviews dataset
2. Train a Naive Bayes classifier using TF-IDF features
3. Fine-tune a BERT transformer model using GPU on Google Colab
4. Save trained models for inference
5. Build a Flask backend to handle user input and predictions
6. Design the frontend using HTML and CSS
7. Integrate confidence scores and accuracy comparison
8. Version control the project and host it on GitHub

---



```
```
