# Subject

The exercise consists of the following tasks which you are asked to complete and submit by the deadline (Monday 19 April 2021).

Task 1 :

This is a sentiment analysis task. Sentiment analysis, in this case, consists of classifying the opinion of a sentence. For this task you will use the Dataset Sent1 described below. You will:

Write a function to read the relevant data (preprocess and clean them as well if needed).

Use the laserembeddings library (https://pypi.org/project/laserembeddings/) to transform each sentence in an embedding array, i.e. to get a numerical representation of each sentence separately.

Implement a classification algorithm which takes these embeddings as input and the values representing the opinions at the output. The choice for the classifier algorithm/system is up to you.

Task 2 :

This is a machine translation task. You are required to use the Dataset Trans1 described below containing sentences in different languages, to train a system that is able to translate language A to language B. The choice of architecture is up to you, but it must contain Recurrent Neural Network layers.

# How to use

```bash
pip install -r requirements.txt
python download_csv.py
python sentiment_analyses.py
```

# Unfortunaly second task isn't working
You'll get an error when starting translation_en_fr.py
