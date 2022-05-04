import json
import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Perceptron
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn import metrics

print("Start 2")
data = pd.read_json('./train.jsonl', lines=True)
print("End 2")

print("Start 3")
docs_train, docs_test, y_train, y_test = train_test_split(
    data.text, data.lang, test_size=0.5)
print("End 3")

print("Start 4")
vectorizer = TfidfVectorizer(ngram_range=(1, 3), analyzer='char',
                             use_idf=False)
print(vectorizer)
print("End 4")

print("Start 5")
clf = Pipeline([
    ('vec', vectorizer),
    ('clf', Perceptron()),
])
print(clf)
print("End 5")

print("Start 6")
clf.fit(docs_train, y_train)
print("End 6")

print("Start 7")
y_predicted = clf.predict(docs_test)
print("End 7")

print("Start 8")
cm = metrics.confusion_matrix(y_test, y_predicted)
print(cm)
print("End 8")

print("Start 12")
dataTest = pd.read_json('./test.jsonl', lines=True)
print("End 12")

print("Start 13")
predicted = clf.predict(dataTest.text)
print("End 13")

print("Start 14")
my_dict = {}
listKey1=[]
listKey2=[]
for s, p in zip(dataTest.id, predicted):
    #print('The language of "%s" is "%s"' % (s, p))
    listKey1.append(s)
    listKey2.append(p)

prediction_result = [{'id': key1, 'lang': key2} for key1, key2 in zip(listKey1, listKey2)]
df = pd.DataFrame(prediction_result)
result = df.to_json(orient="records")
parsed = json.loads(result)

with open('./predictions.jsonl', 'w', encoding='utf-8') as c:
    json.dump(parsed, c, ensure_ascii=False, indent=4)

print("End 14")
