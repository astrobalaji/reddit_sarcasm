
# coding: utf-8
"""
Created on Thu Dec 20 18:41:44 2018

@author: astrobalaji
"""

import pandas as pd
import numpy as np
import spacy as sp
from tqdm import tqdm
from nltk.stem.snowball import SnowballStemmer
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

### TQDM for progress during preprocessing ###
tqdm.pandas()

### loading text corpus for NLP ###
nlp = sp.load("en_core_web_lg")
stemmer = SnowballStemmer('english', ignore_stopwords=True)

get_ipython().config.get('IPKernelApp', {})['parent_appname'] = ""


### Mapped function that converts a string into word vectors through spacy english corpus ####
def col2vec(s):
    slist = str(s).split(" ")
    s= " ".join([stemmer.stem(st) for st in slist])
    return nlp(s).vector

###Loading Training data###
train_df = pd.read_csv("train_10000samps.csv")

train_df.drop(['Unnamed: 0'], axis = 1, inplace = True)

train_df[train_df['label'] == 0].head()

###Prepare vectors to be trained from the comments###
train_vecs = pd.DataFrame()
train_vecs['comment_vecs'] = train_df['comment'].progress_apply(col2vec)
train_vecs['par_comment_vecs'] = train_df['parent_comment'].progress_apply(col2vec)
train_vecs['subreddit'] = train_df['subreddit'].progress_apply(col2vec)
train_vecs['score'] = train_df.score.copy()
train_vecs['ups'] = train_df.ups.copy()
train_vecs['downs'] = train_df.downs.copy()
train_vecs['label'] = train_df.label.copy()
train_x = train_vecs[['comment_vecs', 'par_comment_vecs', 'subreddit', 'score', 'ups', 'downs']]

### Extracting training features and labels ####
for c in train_x.columns:
    if train_x[c].dtype == int:
        train_x[c] = train_x[c].progress_apply(lambda x: [x])
    else:
        train_x[c] = train_x[c].progress_apply(lambda x: list(x))

Train_X = train_x.comment_vecs + train_x.par_comment_vecs + train_x.subreddit + train_x.score + train_x.ups + train_x.downs

X = Train_X.tolist()

X = np.array([np.array(x1) for x1 in X])

Y = train_vecs['label'].astype(float).copy()

### Splitting data into train and test set ###
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.33)


###Fitting the data with a logistic regression model ###
log_reg_mod = LogisticRegression(max_iter=1000, multi_class='multinomial', solver = 'newton-cg', verbose = 10, n_jobs=2).fit(X_train,Y_train)


### Test data ###
model_score = log_reg_mod.score(X_test,Y_test)

print("The model trained with "+str(model_score)+" Accuracy")
