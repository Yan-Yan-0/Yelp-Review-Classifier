import pandas as pd 

data = pd.read_csv('/Users/yanyan/Desktop/MMAI5400/Assignment1/reviews.csv', sep='\t')
data.head()
data.dtypes

sentiment_list = []
for i in range(data.shape[0]):
    if data['RatingValue'][i] >=4:
        sentiment_list.append('Positive')
    elif data['RatingValue'][i] ==3:
        sentiment_list.append('Neutral')
    else:
        sentiment_list.append('Negative')

data['Sentiment'] = sentiment_list
data.head()

data.groupby(['Sentiment']).count()
#use the min value for all group 158 as the baseline 
negative = data[data['Sentiment']=='Negative']
negative.shape

neutral = data[data['Sentiment']=='Neutral'].sample(n=158, random_state=7)
neutral.shape
neutral.head()

positive = data[data['Sentiment']=='Positive'].sample(n=158, random_state=7)
positive.shape
positive.head()

new_data = pd.concat([positive, neutral, negative])
new_data.shape

from sklearn.model_selection import train_test_split
p_trn, p_val = train_test_split(positive, test_size=0.2, random_state=7)
neg_trn, neg_val = train_test_split(negative, test_size=0.2, random_state=7)
neu_trn, neu_val = train_test_split(neutral, test_size=0.2, random_state=7)

train = pd.concat([p_trn, neg_trn, neu_trn])
valid = pd.concat([p_val, neg_val, neu_val])

train.head()

data_train, data_valid = train_test_split(new_data, test_size=0.33, random_state=7)
data_train.shape
data_valid.shape
train.drop(columns='Sentiment', inplace=True)
valid.drop(columns='Sentiment', inplace=True)
train.reset_index(inplace=True)
valid.reset_index(inplace=True)
train.drop(columns='index', inplace=True)
train.to_csv('/Users/yanyan/Desktop/MMAI5400/Assignment2/training.csv', index=False)
valid.drop(columns='index', inplace=True)
valid.to_csv('/Users/yanyan/Desktop/MMAI5400/Assignment2/valid.csv', index=False)

from sklearn.utils import shuffle
train = shuffle(train)
valid = shuffle(valid)