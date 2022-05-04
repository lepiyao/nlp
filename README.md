# nlp
NLP's Assignments

Assignment 1 for Language Identification 

We're using MultinomialNB because we're think it is the fastest one from this testing using Perceptron, KNN with 3 neighbours and MultinomialNB

12:28 perceptron start - 12:32 perceptron done

12:33 MultinomialNB start - 12:35 MultinomialNB done

12:37 KNN 3 neighbors start - 12:45++ KNN 3 neighbors done

How to Use :
  - Place the .py code in the **same folder** as the Train and Test data
  - Train and Test data must be in **.jsonl** file type
  - Train data must be named as **train.jsonl"** and **"test.jsonl"** for Test data
  - Train data must contain these Keys: **lang** (for the Language Type), **text** (for the Sentences)
  - Test data must contain these Keys: **id** (for the ID of the Sentences), **text** (for the Sentences)
  - After the program is finished, it will save the result in **predictions.jsonl** file with the **id** from Test data and **lang** from Train data as the Key
