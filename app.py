from flask import Flask, render_template, request
import pickle
import re
import nltk
import torch

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import BertTokenizer, BertForSequenceClassification

# -------------------------
# NLTK setup
# -------------------------
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# -------------------------
# Flask app
# -------------------------
app = Flask(__name__)

# -------------------------
# Load Naive Bayes model
# -------------------------
nb_model = pickle.load(open("model/sentiment_model.pkl", "rb"))
nb_vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

# -------------------------
# Load BERT model
# -------------------------
bert_tokenizer = BertTokenizer.from_pretrained("bert_model")
bert_model = BertForSequenceClassification.from_pretrained("bert_model")
bert_model.eval()

# -------------------------
# Text preprocessing (NB)
# -------------------------
def preprocess_nb(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    return " ".join(tokens)

# -------------------------
# BERT prediction
# -------------------------
def predict_bert(text):
    inputs = bert_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )
    with torch.no_grad():
        outputs = bert_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        confidence, prediction = torch.max(probs, dim=1)

    return prediction.item(), round(confidence.item() * 100, 2)

# -------------------------
# Routes
# -------------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    sentiment = None
    model_used = "BERT"
    accuracy = None
    confidence_value = None

    # Fixed (training-time) accuracies
    nb_accuracy = "86.0%"
    bert_accuracy = "91.5%"

    if request.method == 'POST':
        text = request.form.get('text')
        model_used = request.form.get('model')

        # ---- Naive Bayes ----
        if model_used == "NB":
            processed = preprocess_nb(text)
            vector = nb_vectorizer.transform([processed])
            prediction = nb_model.predict(vector)[0]
            confidence = max(nb_model.predict_proba(vector)[0]) * 100

            confidence_value = round(confidence, 2)
            accuracy = nb_accuracy

            if prediction == 1:
                sentiment = f"Positive 😊 ({confidence_value}%)"
            else:
                sentiment = f"Negative 😞 ({confidence_value}%)"

        # ---- BERT ----
        else:
            prediction, confidence_value = predict_bert(text)
            accuracy = bert_accuracy

            if prediction == 1:
                sentiment = f"Positive 😊 ({confidence_value}%)"
            else:
                sentiment = f"Negative 😞 ({confidence_value}%)"

    return render_template(
        "index.html",
        sentiment=sentiment,
        model_used=model_used,
        accuracy=accuracy,
        confidence_value=confidence_value,
        nb_accuracy=nb_accuracy,
        bert_accuracy=bert_accuracy
    )

# -------------------------
# Run app
# -------------------------
if __name__ == '__main__':
    app.run(debug=True)
