import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

training = pd.read_csv('/Users/yanyan/Desktop/MMAI5400/Assignment1/training.csv')
valid = pd.read_csv('/Users/yanyan/Desktop/MMAI5400/Assignment1/valid.csv')
#training = pd.read_csv('training.csv')
#valid = pd.read_csv('valid.csv')

#adding sentimental category corresponding to rating value 
#define a function to add efficiency 
def add_sentiment(data): 
    sentiment_list = []
    for i in range(data.shape[0]):
        if data['RatingValue'][i] >=4:
            sentiment_list.append('Positive')
        elif data['RatingValue'][i] ==3:
            sentiment_list.append('Neutral')
        else:
            sentiment_list.append('Negative')
    data['Sentiment'] = sentiment_list

add_sentiment(training)
add_sentiment(valid)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
#build a pipeline for faster calculation
from sklearn.pipeline import Pipeline
text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
])

#fit the pipeline model 
text_clf.fit(training.Review, training.Sentiment)

#test for accuracy using validaton dataset 'valid'
prediction = text_clf.predict(valid.Review)

from sklearn import metrics
accuracy=metrics.accuracy_score(valid.Sentiment, prediction)
print('accuracy: '+ str(accuracy))
f1=metrics.f1_score(valid.Sentiment, prediction, average='weighted')
print('F1_score: '+ str(f1))
metrics.confusion_matrix(valid.Sentiment, prediction)

metrics.plot_confusion_matrix(text_clf, valid.Sentiment, prediction, display_labels=['negative', 'neutral', 'positive'])
print('Confusion_matrix:')
plt.show()

metrics.confusion_matrix(valid.Sentiment, prediction)

