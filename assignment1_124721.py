"""
Group Nekomata
Levi Hanny Santoso - 124721
Mohammad Seyedalizadeh - 124744
"""

import json
import pandas as pd 
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report

"""
From https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
Read json data 
"""
data = pd.read_json('./train.jsonl', lines=True)
# Read the test file
testData = pd.read_json('./test.jsonl', lines=True)

#Set data to train and test with splitting the data
docs_train, docs_test, y_train, y_test = train_test_split(
    data.text, data.lang, test_size=0.5)

"""
Use TfidfVectorizer to shorten the code length from 
using CountVectorizer followed by TfidfTransformer
(But I think it will took less time if we use CountVectorizer followed by TfidfTransformer)
Set ngram_range from unigrams to trigrams
"""
vectorizer = TfidfVectorizer(ngram_range=(1, 3), analyzer='char', use_idf=False)

"""
From https://www.youtube.com/watch?v=0B5eIE_1vpU&t=1411s
Create the Classifier using the vectorizer
"""
clf = Pipeline([
    ('vec', vectorizer),
    ('clf', MultinomialNB()),
])

# Fit the pipeline to training set
clf.fit(docs_train, y_train)

# Predict the outcome using the testing data that has been split previously
splitPredicted = clf.predict(docs_test)

#Printing Confusion Matrix and the classification report based on the split data testing
cm = metrics.confusion_matrix(y_test, splitPredicted)
print(cm)
print(classification_report(y_test, splitPredicted, target_names=np.unique(y_train)))

# Predict the outcome using the Test data using the .jsonl file
predicted = clf.predict(testData.text)


"""
Creating Dataframe from the id and lang
Then using pandas to conver from Data Frame to .jsonl file
From https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_json.html
 & https://pandas.pydata.org/docs/reference/api/pandas.concat.html
"""
df = pd.concat([testData.id, pd.DataFrame(predicted, columns=['lang'])], axis=1)
df.to_json('predictions.jsonl', orient='records', lines=True)
