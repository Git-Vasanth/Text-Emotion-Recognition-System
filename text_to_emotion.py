import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud

#Reading Data
data = pd.read_csv('train.txt' , header = None , sep = ';' , names = ['Comment' , 'Emotion'])

#Renaming Columns
data['length'] = [len(x) for x in data['Comment']]

#checking Null Values
data.isnull().sum()

#Checking Duplicates Value
data.duplicated().sum()

#Value Counts of cooments associated with emotion
data['Emotion'].value_counts()

emotions_list = data['Emotion'].unique()

plt.figure(figsize=(8, 6))
sns.countplot(x=data['Emotion'])
plt.title("Count of Comments by Emotion")
plt.xlabel("Emotion")
plt.ylabel("Count")
plt.show()

# Plotting multiple stacked histograms for length of comments by emotion
plt.figure(figsize=(10, 6))
sns.histplot(data=data, x='length', hue='Emotion', multiple='stack')
plt.title("Distribution of Comment Length by Emotion")
plt.xlabel("Comment Length")
plt.ylabel("Frequency")
plt.show()

# Plotting word clouds for each emotion
for emotion in emotions_list:
    text = " ".join(data.loc[data['Emotion'] == emotion, 'Comment'])
    wc = WordCloud(width=700, height=700).generate(text)
    plt.figure(figsize=(10, 10))
    plt.imshow(wc, interpolation='bilinear')
    plt.title(emotion + " Word Cloud")
    plt.axis('off')
    plt.show()

from sklearn.preprocessing import LabelEncoder
label_enc = LabelEncoder()
data['num_labels'] = label_enc.fit_transform(data['Emotion'])

df = data.copy()

#Applying ML
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer

#metrics
from sklearn.metrics import accuracy_score , classification_report

import nltk
import re

stopwords = set(nltk.corpus.stopwords.words('english'))

from nltk.stem import PorterStemmer
porter = PorterStemmer()


def clean_and_stem(text):
    # Define a regex pattern to match non-alphabetic characters
    pattern = r'[^a-zA-Z\s]'

    cleaned_text = re.sub(pattern, '', text)

    words = cleaned_text.split()

    porter = PorterStemmer()

    stemmed_words = [porter.stem(word) for word in words]

    stemmed_text = " ".join(stemmed_words)

    return stemmed_text

df['clean_text'] = df['Comment'].apply(clean_and_stem)

x = df['clean_text']
y = df['num_labels']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

tfidf = TfidfVectorizer()

x_train_tfidf = tfidf.fit_transform(x_train)
x_test_tfidf = tfidf.transform(x_test)

clfs = {
    "MNB": MultinomialNB(),
    "LG": LogisticRegression(),
    "RFC": RandomForestClassifier(),
    "SVM": SVC()
}

# Train and evaluate each classifier
for name, clf in clfs.items():
    print(f"Training {name}...")

    clf.fit(x_train_tfidf, y_train)

    y_pred = clf.predict(x_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy Score for {name}: {accuracy}")

    print(f"Classification Report for {name}:")
    print(classification_report(y_test, y_pred))

rf = RandomForestClassifier()
rf.fit(x_train_tfidf, y_train)
rf_y_pred = rf.predict(x_test_tfidf)

additional_sentences = [
    "Iam Scared and disappointed"
]

def predict_emotion(input_text):
    cleaned_text = clean_and_stem(input_text)
    input_vectorized = tfidf.transform([cleaned_text])

    predicted_label = rf.predict(input_vectorized)[0]
    predicted_emotion = label_enc.inverse_transform([predicted_label])[0]
    label = np.max(rf.predict_proba(input_vectorized))

    return predicted_emotion, label

for sentence in additional_sentences:
    print(sentence)
    pred_emotion, label = predict_emotion(sentence)
    print("Prediction :", pred_emotion)
    print("Label (Probability) :", label)
    print("================================================================")
