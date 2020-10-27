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


#tokenizing text 
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
train_counts = count_vect.fit_transform(training.Review)
train_counts.shape

#get the word frequency and downscaling the weight of less informative words 
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
train_tfidf = tfidf_transformer.fit_transform(train_counts)
train_tfidf.shape

#training the classifer 
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(train_tfidf, training.Sentiment)


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
accuracy = np.mean(prediction == valid.Sentiment)
print('accuracy: '+ str(accuracy))

from sklearn.linear_model import SGDClassifier
text_clf2 = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(loss='hinge', penalty='l2',
                          alpha=1e-3, random_state=42,
                          max_iter=5, tol=None)),
])

text_clf2.fit(training.Review, training.Sentiment)

predicted = text_clf2.predict(valid.Review)

from sklearn import metrics
accuracy=metrics.accuracy_score(valid.Sentiment, predicted)
print('accuracy: '+ str(accuracy))
f1=metrics.f1_score(valid.Sentiment, predicted, average='weighted')
print('F1_score: '+ str(f1))
metrics.plot_confusion_matrix(text_clf2, valid.Sentiment, predicted, display_labels=['negative', 'neutral', 'positive'])
plt.show()