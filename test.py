import re
import nltk
import string
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Modelling
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, LSTM, Embedding, Bidirectional

# Modelling
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.svm import SVC

lemmatizer = WordNetLemmatizer()
# %%

# CSV dosyasının bulunduğu yol veya URL
train_file_path = '/Users/musti/Desktop/AIfinalProject/emotionsdata_train.csv'
# test_file_path = '/Users/musti/Desktop/AIfinalProject/emotionsdata_test..csv'

# CSV dosyasını oku ve bir DataFrame'e yükle
train_data = pd.read_csv(train_file_path, names=['Text', 'Emotion'], sep=',')
# test_data = pd.read_csv(test_file_path, names=['Text', 'Emotion'], sep=',')

# Sütun isimlerini düzenle
train_data.columns = ['Text', 'Emotion']
# test_data.columns = ['Text', 'Emotion']

# İlk 5 veriyi gösterme
print(train_data.head())
# print(test_data.head())

# Satır ve Stün sayılarını gösteriyor
print(train_data.shape)
# print(train_data.shape)
# %%
# Null veya Tekrarlanan değerli gösterme
print(train_data.isnull().sum())
print(train_data.duplicated().sum())

# Null veya Tekrarlanan değerli gösterme
# print(test_data.isnull().sum())
# print( test_data.duplicated().sum())
# %%
# Hangi özellikten kaç adet veri olduğunu gösteriyor
train_data.Emotion.value_counts()
# test_data.Emotion.value_counts()

# %%
import pandas as pd
import re


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


def remove_small_sentences(df):
    for i in range(len(df)):
        if len(df.text.iloc[i].split()) < 3:
            df.text.iloc[i] = np.nan


def normalize_text(df):
    df['Text'] = df['Text'].apply(lower_case)
    df['Text'] = df['Text'].apply(remove_numbers)
    df['Text'] = df['Text'].apply(remove_punctuations)
    df['Text'] = df['Text'].apply(lemmatization)
    return df


def normalized_sentence(sentence):
    sentence = lower_case(sentence)
    sentence = remove_numbers(sentence)
    sentence = remove_punctuations(sentence)
    sentence = lemmatization(sentence)
    return sentence


# %%
# import nltk
# nltk.download('wordnet')

# %%
trainDataResult = normalize_text(train_data)
# testDataResult = normalize_text(test_data)

# %%
print(trainDataResult.head())
# print(testDataResult.head())

# %%
# Verileri test ve train olarak yüzde 80 e 20 şeklinde ayırma

X = trainDataResult['Text']
y = trainDataResult['Emotion']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
print("Eğitim veri seti boyutu (X_train):", X_train.shape)
print("Eğitim veri seti boyutu (y_train):", y_train.shape)
print("Test veri seti boyutu (X_test):", X_test.shape)
print("Test veri seti boyutu (y_test):", y_test.shape)
# %%
# Verilerin dengeli olup olmadığını kontrol etme

trainDataResult.Emotion.value_counts() / trainDataResult.shape[0] * 100

# %%
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4))
sns.countplot(x='Emotion', data=trainDataResult);


# %%
def train_model(model, data, targets):
    """
    Train a model on the given data and targets.

    Parameters:
    model (sklearn model): The model to be trained.
    data (list of str): The input data.
    targets (list of str): The targets.

    Returns:
    Pipeline: The trained model as a Pipeline object.
    """
    # Create a Pipeline object with a TfidfVectorizer and the given model
    text_clf = Pipeline([('vect', TfidfVectorizer()),
                         ('clf', model)])
    # Fit the model on the data and targets
    text_clf.fit(data, targets)
    return text_clf


# %%
def get_F1(trained_model, X, y):
    """
    Get the F1 score for the given model on the given data and targets.

    Parameters:
    trained_model (sklearn model): The trained model.
    X (list of str): The input data.
    y (list of str): The targets.

    Returns:
    array: The F1 score for each class.
    """
    # Make predictions on the input data using the trained model
    predicted = trained_model.predict(X)
    # Calculate the F1 score for the predictions
    f1 = f1_score(y, predicted, average=None)
    # Return the F1 score
    return f1


# %%
log_reg = train_model(LogisticRegression(solver='liblinear', random_state=0), X_train, y_train)

# %%
y_pred = log_reg.predict(['yarın okulda ödevim var'])
y_pred
# %%
y_pred = log_reg.predict(X_test)

# calculate the accuracy
log_reg_accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: ', log_reg_accuracy, '\n')

# calculate the F1 score
f1_Score = get_F1(log_reg, X_test, y_test)
pd.DataFrame(f1_Score, index=trainDataResult.Emotion.unique(), columns=['F1 score'])

# %%
##Classification Report
print(classification_report(y_test, y_pred))

# %%
# Train the model with the training data
DT = train_model(DecisionTreeClassifier(random_state=0), X_train, y_train)

# test the model with the test data
y_pred = DT.predict(X_test)

# calculate the accuracy
DT_accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: ', DT_accuracy, '\n')

# calculate the F1 score
f1_Score = get_F1(DT, X_test, y_test)
pd.DataFrame(f1_Score, index=trainDataResult.Emotion.unique(), columns=['F1 score'])
# %%
##Classification Report
print(classification_report(y_test, y_pred))
# %%
# Train the model with the training data
SVM = train_model(SVC(random_state=0), X_train, y_train)

# test the model with the test data
y_pred = SVM.predict(X_test)

# calculate the accuracy
SVM_accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: ', SVM_accuracy, '\n')

# calculate the F1 score
f1_Score = get_F1(SVM, X_test, y_test)
pd.DataFrame(f1_Score, index=trainDataResult.Emotion.unique(), columns=['F1 score'])
# %%
##Classification Report
print(classification_report(y_test, y_pred))

# %%
# Train the model with the training data
RF = train_model(RandomForestClassifier(random_state=0), X_train, y_train)

# test the model with the test data
y_pred = RF.predict(X_test)

# calculate the accuracy
RF_accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: ', RF_accuracy, '\n')

# calculate the F1 score
f1_Score = get_F1(RF, X_test, y_test)
pd.DataFrame(f1_Score, index=trainDataResult.Emotion.unique(), columns=['F1 score'])
# %%
##Classification Report
print(classification_report(y_test, y_pred))
# %%
models = pd.DataFrame({
    'Model': ['Logistic Regression', 'Decision Tree', 'Support Vector Machine', 'Random Forest'],
    'Accuracy': [round(log_reg_accuracy, 2), round(DT_accuracy, 2), round(SVM_accuracy, 2), round(RF_accuracy, 2)]
})

models.sort_values(by='Accuracy', ascending=False).reset_index().drop(['index'], axis=1)