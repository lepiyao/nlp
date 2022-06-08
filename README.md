# Language Identification
NLP's Assignments

**Group Nekomata** - Assignment 1 for Language Identification

We're using MultinomialNB because we're think it is the fastest one from this testing using Perceptron, KNN with 3 neighbours and MultinomialNB

12:28 Perceptron start - 12:32 Perceptron done

12:33 MultinomialNB start - 12:35 MultinomialNB done

12:37 KNN 3 neighbors start - 12:45++ KNN 3 neighbors done

How to Use :
  - Place the .py code in the **same folder** as the Train and Test data
  - Train and Test data must be in **.jsonl** file type
  - Train data must be named as **train.jsonl"** and **"test.jsonl"** for Test data
  - Train data must contain these Keys: **lang** (for the Language Type), **text** (for the Sentences)
  - Test data must contain these Keys: **id** (for the ID of the Sentences), **text** (for the Sentences)
  - After the program is finished, it will save the result in **predictions.jsonl** file with the **id** from Test data and **lang** from Train data as the Key



# Part-of-Speech Tagging
NLP's Assignments

Assignment 3 for Part-of-Speech Tagging
**Group Nekomata** 
- Levi Hanny Santoso - 124721
- Mohammad Seyedalizadeh - 124744

How to Use :
  - Place the .py code in the **same folder** as the **training.conllu**, **dev.conllu**, **test.conllu** and **base_config.cfg**
  - First we need to make a **config.cfg** using CLI by running this ```python -m spacy init fill-config base_config.cfg config.cfg```
  - Then we need to change the **training** and **dev** from **.conllu** file type to **.spacy** file type by using CLI
	 
      ```python -m spacy convert **training.conllu** -n 10 **destination_folder**``` 
   
      and 
   
      ```python -m spacy convert **dev.conllu** -n 10 **destination_folder**```
   
  - Then you can choose, do you want to train the data using Python Script or using CLI
       
       **For CLI**
       - You can run this script ```python -m spacy train config.cfg --output ./**..\model** --paths.train ./**training.spacy** --paths.dev ./**dev.spacy**```
       - Then wait until it finish and there is a new folder called **model**
       - Then run the **.py**, and the result will be written in **test.conllu**
       
       **For Python**
       - Remove the Command on the **#11** line
       - Then you can run the **.py** file and it will automate the process
       - Then run the **.py**, and the result will be written in **test.conllu**
       
