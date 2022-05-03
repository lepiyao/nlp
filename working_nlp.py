#!/usr/bin/env python
# coding: utf-8

# In[1]:
import json
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Perceptron
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn import metrics


# In[2]:

print("Start 2")
data = pd.read_json('./train.jsonl', lines=True)
print("End 2")

# In[3]:

print("Start 3")
docs_train, docs_test, y_train, y_test = train_test_split(
    data.text, data.lang, test_size=0.5)
print("End 3")

# In[4]:

print("Start 4")
vectorizer = TfidfVectorizer(ngram_range=(1, 3), analyzer='char',
                             use_idf=False)
print("End 4")

# In[5]:

print("Start 5")
clf = Pipeline([
    ('vec', vectorizer),
    ('clf', Perceptron()),
])
print("End 5")

# In[6]:

print("Start 6")
clf.fit(docs_train, y_train)
print("End 6")

# In[7]:

print("Start 7")
y_predicted = clf.predict(docs_test)
print("End 7")

# In[12]:

print("Start 8")
cm = metrics.confusion_matrix(y_test, y_predicted)
print(cm)
print("End 8")

# In[34]:

print("Start 9")
sentences = [
    'Την περίοδο 2005-06 ο Άρης αγωνίστηκε στην Β ́ Εθνική.',
    'En Europa Central se enseñan como primeras lenguas alemán, polaco, rumano, checo, húngaro, serbio, esloveno, croata, eslovaco, y luxemburgués.',
    'Следует помнить, что случаи излечения кандидозной инфекции одними лишь народными средствами официальной медицине неизвестны',
]
print("End 9")

# In[35]:

print("Start 10")
predicted = clf.predict(sentences)
print("End 10")

# In[49]:

print("Start 11")
for s, p in zip(sentences, predicted):
    i = 0
    print('The language of "%s" is "%s"' % (s, p))
print("End 11")

# In[52]:

print("Start 12")
dataTest = pd.read_json('./test.jsonl', lines=True)
print("End 12")

# In[54]:

print("Start 13")
predicted = clf.predict(dataTest.text)
print("End 13")

# In[57]:

print("Start 14")
my_dict = {}
listKey1=[]
listKey2=[]
for s, p in zip(dataTest.id, predicted):
    #print('The language of "%s" is "%s"' % (s, p))
    listKey1.append(s)
    listKey2.append(p)

#https://stackoverflow.com/questions/57475199/python-combine-multiple-lists-into-one-json-array
datas = {'prediction_result': [{'id': key1, 'lang': key2} for key1, key2 in zip(listKey1, listKey2)]}

with open('./predictions.jsonl', 'w', encoding='utf-8') as f:
    json.dump(datas, f, ensure_ascii=False, indent=4)
print("End 14")
