import os
import re
import numpy as np
import pandas as pd
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report
import uvicorn
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List

from starlette.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates

app = FastAPI()

# Şablonlar için Jinja2Templates yükleyin
templates = Jinja2Templates(directory="templates")

# Statik dosyalar için StaticFiles yükleyin
app.mount("/static", StaticFiles(directory="static"), name="static")

# Ana sayfa uç noktası
@app.get("/")
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# CSV dosyasının bulunduğu yol veya URL
base_dir = '/Users/musti/Desktop/Projects/Python/AIfinalProject/'
train_file_path = os.path.join(base_dir, 'emotionsdata_train.csv')
test_file_path = os.path.join(base_dir, 'emotionsdata_test.csv')

# Dosyaların varlığını kontrol et
if not os.path.exists(train_file_path):
    raise FileNotFoundError(f"Train file not found: {train_file_path}")
if not os.path.exists(test_file_path):
    raise FileNotFoundError(f"Test file not found: {test_file_path}")

# CSV dosyasını oku ve bir DataFrame'e yükle
train_data = pd.read_csv(train_file_path, names=['Text', 'Emotion'], sep=',')
test_data = pd.read_csv(test_file_path, names=['Text', 'Emotion'], sep=',')

train_data.columns = ['Text', 'Emotion']
test_data.columns = ['Text', 'Emotion']

def lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    text = text.split()
    text = [lemmatizer.lemmatize(y) for y in text]
    return " ".join(text)

def remove_numbers(text):
    text = ''.join([i for i in text if not i.isdigit()])
    return text

def lower_case(text):
    text = text.split()
    text = [y.lower() for y in text]
    return " ".join(text)

def remove_punctuations(text):
    text = re.sub(r'[%\s!"#$%&\'()*+,،.-/:;<=>؟?@[\]^_`{|}~]', ' ', text)
    text = text.replace('؛', "", )
    text = re.sub(r'\s+', ' ', text)
    text = " ".join(text.split())
    return text.strip()

def normalize_text(df):
    df['Text'] = df['Text'].apply(lower_case)
    df['Text'] = df['Text'].apply(remove_numbers)
    df['Text'] = df['Text'].apply(remove_punctuations)
    df['Text'] = df['Text'].apply(lemmatization)
    return df

trainDataResult = normalize_text(train_data)
testDataResult = normalize_text(test_data)

X = trainDataResult['Text']
y = trainDataResult['Emotion']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(model, data, targets):
    text_clf = Pipeline([('vect', TfidfVectorizer()), ('clf', model)])
    text_clf.fit(data, targets)
    return text_clf

def get_F1(trained_model, X, y):
    predicted = trained_model.predict(X)
    f1 = f1_score(y, predicted, average=None)
    return f1

# Model eğitimleri
log_reg = train_model(LogisticRegression(solver='liblinear', random_state=0), X_train, y_train)
DT = train_model(DecisionTreeClassifier(random_state=0), X_train, y_train)
SVM = train_model(SVC(random_state=0), X_train, y_train)
RF = train_model(RandomForestClassifier(random_state=0), X_train, y_train)

# API için veri modeli
class TextInput(BaseModel):
    texts: List[str]

# Uç noktalar
@app.post("/predict/logistic_regression")
def predict_logistic_regression(input: TextInput):
    predictions = log_reg.predict(input.texts)
    return {"predictions": predictions.tolist()}

@app.post("/predict/decision_tree")
def predict_decision_tree(input: TextInput):
    predictions = DT.predict(input.texts)
    return {"predictions": predictions.tolist()}

@app.post("/predict/svm")
def predict_svm(input: TextInput):
    predictions = SVM.predict(input.texts)
    return {"predictions": predictions.tolist()}

@app.post("/predict/random_forest")
def predict_random_forest(input: TextInput):
    predictions = RF.predict(input.texts)
    return {"predictions": predictions.tolist()}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
