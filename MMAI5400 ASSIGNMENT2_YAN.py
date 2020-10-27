import pandas as pd
import nltk
from nltk.corpus import stopwords
import fastai
from fastai import *
import fastai.text as ftx
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt


training_file = '/Users/yanyan/Desktop/MMAI5400/Assignment2/training.csv'
valid_file = '/Users/yanyan/Desktop/MMAI5400/Assignment2/valid.csv'
training = pd.read_csv(training_file)
valid = pd.read_csv(valid_file)

# adding sentimental category corresponding to rating value
# define a function to add efficiency


def add_sentiment_value(data):
    sentiment_list = []
    for i in range(data.shape[0]):
        if data['RatingValue'][i] < 3:
            sentiment_list.append(0)
        elif data['RatingValue'][i] == 3:
            sentiment_list.append(1)
        else:
            sentiment_list.append(2)
    data['sentiment_value'] = sentiment_list


add_sentiment_value(training)
add_sentiment_value(valid)

# Preprocessing for both training and valid dataset

training['Review'] = training['Review'].str.replace("[^a-zA-Z]", " ")
valid['Review'] = valid['Review'].str.replace("[^a-zA-Z]", " ")

# Tokenize
tokenized_train = training['Review'].apply(lambda x: x.split())
tokenized_valid = valid['Review'].apply(lambda x: x.split())

# Remove stop-words
stop_words = stopwords.words('english')
tokenized_train = tokenized_train.apply(lambda x: [item for item in x if item
                                        not in stop_words])
tokenized_valid = tokenized_valid.apply(lambda x: [item for item in x if item
                                        not in stop_words])

# De-tokenization
detokenized_train = []
for i in range(len(training)):
    t = ' '.join(tokenized_train[i])
    detokenized_train.append(t)

training['Review'] = detokenized_train

detokenized_valid = []
for i in range(len(valid)):
    t = ' '.join(tokenized_valid[i])
    detokenized_valid.append(t)

valid['Review'] = detokenized_valid

# Select the data to train and valid
training_model = training[['sentiment_value', 'Review']]
valid_model = valid[['sentiment_value', 'Review']]

# Assemble the data into fast.ai's Text<LM/Class>DataBunch
# Language model data
data_lm = ftx.TextLMDataBunch.from_df(train_df=training_model,
                                      valid_df=valid_model, path="")

# Classifier model data
data_clas = ftx.TextClasDataBunch.from_df(path="", train_df=training_model,
                                          valid_df=valid_model,
                                          vocab=data_lm.train_ds.vocab, bs=64)

# Fine-Tuning the Pre-Trained Model and Making Predictions
learn = ftx.language_model_learner(data_lm, ftx.AWD_LSTM, drop_mult=0.5)


# ULMfit
learn.unfreeze()
moms = (0.8, 0.7)
learn.fit_one_cycle(cyc_len=4, max_lr=slice(1e-2), moms=moms)
learn.save_encoder('enc')

# Train only the classification head
learn = ftx.text_classifier_learner(data_clas, ftx.AWD_LSTM)
learn.load_encoder('enc')
learn.fit_one_cycle(4, slice(2e-3), moms=moms)
learn.save('stage0-clf')

learn.freeze_to(-2)
learn.fit_one_cycle(4, slice(1e-3, 2e-3), moms=moms)

learn.save('stage1-clf')
# Fine-tune the encoder together with a classification head for slightly
# better performance.
learn.unfreeze()
learn.fit_one_cycle(4, slice(1e-3/2, 2e-3), moms=moms)
learn.save('final_model')
learn.export('/Users/yanyan/Desktop/MMAI5400/Assignment2/final_model.kpl')

# Evaluate the result
preds, targets, losses = learn.get_preds(with_loss=True)
predictions = np.argmax(preds, axis=1)

accuracy = metrics.accuracy_score(targets, predictions)
print('accuracy: ' + str(accuracy))

f1 = metrics.f1_score(targets, predictions, average='weighted')
print('F1_score: ' + str(f1))

print('Confusion_matrix: ')
print(pd.crosstab(predictions, targets))
interp = fastai.train.ClassificationInterpretation(learn, preds, targets,
                                                   losses)
interp.plot_confusion_matrix()
plt.show()