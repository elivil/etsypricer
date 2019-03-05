# coding: utf-8
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from tensorflow.keras.applications import mobilenet_v2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import logging
import os
import tempfile

df = pd.read_json('cleandf.json')
len(df)
train_texts = [text.split() for text in df['adesc'].values]
len(train_texts)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

TEMP_FOLDER = "PATH"
print('Folder "{}" will be used to save temporary dictionary and corpus.'.format(TEMP_FOLDER))
# # VECTOR SPACE MODEL #
from sklearn.preprocessing import normalize
def normvec(vec):
    normv = normalize(vec[:,np.newaxis], axis=0).ravel()
    return normv
normvec(np.array([1,1]))
# ## TFIDF ##
from gensim import corpora
dictionary = corpora.Dictionary(train_texts)
dictionary.save(os.path.join(TEMP_FOLDER, 'necklaces.dict'))  # store the dictionary, for future reference
#dictionary = corpora.dictionary.Dictionary.load(os.path.join(TEMP_FOLDER, 'necklaces.dict'))
print(dictionary)
corpus = [dictionary.doc2bow(text) for text in train_texts]
corpora.MmCorpus.serialize(os.path.join(TEMP_FOLDER, 'necklaces.mm'), corpus)
#corpus = corpora.MmCorpus(os.path.join(TEMP_FOLDER, 'necklaces.mm'))
from gensim import models
tfidf = models.TfidfModel(corpus) # step 1 -- initialize a model
tfidf.save(os.path.join(TEMP_FOLDER, 'necklacesmedmodel.tfidf'))
from gensim import similarities
index = similarities.MatrixSimilarity(tfidf[corpus]) # transform corpus to Tfidf space and index it
index.save(os.path.join(TEMP_FOLDER, 'necklacesmedtfidfsim.index'))
corpus_test = [dictionary.doc2bow(text) for text in test_texts]
corpora.MmCorpus.serialize(os.path.join(TEMP_FOLDER, 'necklacesmedtest.mm'), corpus_test)
Xtraintfidf = tfidf[corpus]
Xtesttfidf = tfidf[corpus_test]
# ## DOC2VEC
import gensim
def read_corpus(texts, tokens_only=False):
    for i, text in enumerate(texts):
        if tokens_only:
            yield gensim.utils.simple_preprocess(' '.join(text))
        else:
            yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(' '.join(text)), [i])
train_corpus = list(read_corpus(train_texts))
#test_corpus = list(read_corpus(test_texts, tokens_only=True))
get_ipython().run_cell_magic('time', '', "model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=40,dm=0, dbow_words=1)\n\nmodel.build_vocab(train_corpus)\n\nmodel.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)\nmodel.save(os.path.join(TEMP_FOLDER, 'necklaces.d2v'))")
get_ipython().run_line_magic('time', 'X_tr_d2v = np.array([model.infer_vector(train_corpus[i].words, steps=40, alpha=0.025) for i in range(len(train_corpus))])')
#X_test_d2v = np.array([model.infer_vector(test_corpus[i], steps=40, alpha=0.025) for i in range(len(test_corpus))])
np.save('d2v-features',X_tr_d2v)
X_tr_d2v.shape
from sklearn.metrics.pairwise import cosine_similarity
get_ipython().run_cell_magic('time', '', 'd2v_sims = cosine_similarity(X_tr_d2v)')
np.save('d2v-cossim',d2v_sims)
# # Getting Image Features
import tensorflow as tf
# Reads an image from a file, decodes it into a dense tensor, and resizes it
# to a fixed shape.
def _parse_function(filename):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string)
    image_resized = tf.image.resize_images(image_decoded, [224, 224])
    return image_resized
# A vector of filenames.
filenames = tf.constant([os.path.join(PATH,x[0]['path']) for x in data['images'].values])

def create_dataset(filenames, batch_size=1):
    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    dataset = dataset.map(_parse_function)
    dataset = dataset.batch(batch_size)  # Batch size to use
    iterator = dataset.make_one_shot_iterator()
    batch_features = iterator.get_next()
    return batch_features


# define the CNN network
# Here we are using MobileNetV2 and initialising it
# with pretrained imagenet weights
model = mobilenet_v2.MobileNetV2(weights='imagenet',include_top=False,pooling='avg')
features = np.zeros((len(data),1280))
features.shape
get_ipython().run_cell_magic('time', '', 'batch_size=20\ni=0\nwhile (i+batch_size)<len(data):\n    next_batch = create_dataset(filenames[i:i+20], batch_size=batch_size)\n    with tf.Session() as sess:\n            first_batch = sess.run(next_batch)\n            images = preprocess_input(first_batch)\n    features[i:i+batch_size,:] = model.predict(images)\n    i+=batch_size')
features[0]
features[:41]
get_ipython().run_cell_magic('time', '', 'i=0\nfeatures[i:i+20,:] = model.predict(images)')
features.shape
# Model fitting and hyperparameter tuning #
def score_r2(y_pred, y_act):
    assert(len(y_pred)==len(y_act))
    resid = y_pred-y_act
    ssresid = np.sum(np.square(resid))
    sstot = np.sum(np.square(y_act-np.mean(y_act)))
    r2 = 1-ssresid/sstot
    return r2
def rmse(y_pred, y_act):
    assert(len(y_pred)==len(y_act))
    return np.sqrt(np.sum(np.square(y_pred-y_act))/len(y_act))
def mape(y_pred, y_act):
    assert(len(y_pred)==len(y_act))
    return 100*np.sum(np.abs((y_pred-y_act)/y_act))/len(y_act)
def adj_r2(y_est,y_act,dfest,dftot):
    assert(len(y_est)==len(y_act))
    resid = y_est-y_act
    ssresid = np.sum(np.square(resid))
    sstot = np.sum(np.square(y_act-np.mean(y_act)))
    r2adj = 1-ssresid*dftot/sstot/dfest
    return r2adj
# ## kNN regression ##
def knn_predict_tfidf(k, X, y_train, weighting=False):
    y_pred =  np.zeros(len(X))
    for i in range(len(X)):
        sims = sorted(enumerate(index[X[i]]), key=lambda item: -item[1])[:k]
        if weighting:
            weights = np.square(normvec(np.array([v for (k,v) in sims])))
            if np.array_equal(weights,np.zeros(len(weights))):
                weights=None
            y_pred[i] = np.average([y_train[k] for (k,v) in sims], weights=weights)
        else:
            y_pred[i] = np.average([y_train[k] for (k,v) in sims])
    return y_pred
def hyperparamcvknn_tfidf(klist, X, y, y_train):
    results = []
    for k in klist:
        for weighting in [True]:
            y_pred = knn_predict_tfidf(k, X, y_train, weighting)
            results.append((adj_r2(y_pred, y, len(y)/k,len(y)-1),[k,weighting]))
    return results
y_fit = knn_predict_tfidf(10,Xtraintfidf,y_train,True)
print(adj_r2(y_fit,y_train, len(y_train)/10,len(y_train)-1))
np.median(y_train)
rmse(y_fit,y_train)
y_pred = knn_predict_tfidf(10,Xtesttfidf,y_train,True)
print(adj_r2(y_pred,y_test, len(y_test)/10,len(y_test)-1))
np.median(y_test)
rmse(y_pred,y_test)
mape(y_fit,y_train)
mape(y_pred,y_test)
def score_testdata(X_test,y_test,y_train,params):
    y_pred = knn_predict(params['k'],X_test,y_train,params['weighting'])
    return score_r2(y_pred,y_test)
params = {'k': 10, 'weighting': True}    
print(score_testdata(Xtesttfidf, y_test,y_train,params))
get_ipython().run_cell_magic('time', '', "def rmse_testdata(X_test,y_test,y_train,params):\n    y_pred = knn_predict(params['k'],X_test,y_train,params['weighting'])\n    return rmse(y_pred,y_test), y_pred\n                   \nparams = {'k': 10, 'weighting': True}    \nrmse_test, y_predix = rmse_testdata(Xtesttfidf, y_test,y_train,params)\nprint(rmse_test)")
def adj_rsquare(y_pred,y_act,N,k):
    assert(len(y_pred)==len(y_act))
    resid = y_pred-y_act
    ssresid = np.sum(np.square(resid))
    sstot = np.sum(np.square(y_act-np.mean(y_act)))
    dftot = N-1
    dfest = N/k
    r2adj = 1-ssresid*dftot/sstot/dfest
    return r2adj
y_fit = knn_predict(10,Xtraintfidf, y_train,weighting=True)
adj_rsquare(y_fit,y_train,len(y_train),10)
mape(y_predix,y_test)
l=[]
for i in range(10000):
    sample= usdf.sample(n=10, random_state=i)
    l.append(np.mean(sample['price'].values))
np.random.randint(6000)
begin = np.random.randint(6000)
rmse(l[begin:(begin+len(y_test))],y_test)
sns.distplot(l)
np.median(l)
def mape_testdata(X_test,y_test,y_train,params):
    y_pred = knn_predict(params['k'],X_test,y_train,params['weighting'])
    return mape(y_pred,y_test)
params = {'k': 10, 'weighting': True}    
print(mape_testdata(Xtesttfidf, y_test,y_train,params))
X_tr.shape
y_train.shape
# ## word2vec knn
def knn_predict_w2v(k, X, y_train, weighting=False):
    y_pred =  np.zeros(len(X))
    for i in range(len(X)):
        #inferred_vector = model.infer_vector(X[i])
        sims = model.docvecs.most_similar([X[i]], topn=k+1)[1:]
        if weighting:
            weights = np.square(normvec(np.array([v for (k,v) in sims])))
            if np.array_equal(weights,np.zeros(len(weights))):
                weights=None
            y_pred[i] = np.average([y_train[k] for (k,v) in sims], weights=weights)
        else:
            y_pred[i] = np.average([y_train[k] for (k,v) in sims])
    return y_pred
def hyperparamcvknn_w2v(klist, X, y_act, y_train):
    results = []
    for k in klist:
        for weighting in [True,False]:
            y_pred = knn_predict_w2v(k, X, y_train, weighting)
            results.append((rmse(y_pred, y_act),[k,weighting]))
    return results
get_ipython().run_cell_magic('time', '', 'klist=[3,5,7,10,15]\nw2v_hyperparam = hyperparamcvknn_w2v(klist, X_tr_d2v, y_train, y_train)')
w2v_hyperparam
y_pred_d2v = knn_predict_w2v(7,X_test_d2v, y_train,weighting=True)
rmse(y_pred_d2v,y_test)
# ## Random Forest ##
from sklearn.decomposition import TruncatedSVD
from time import time
# Dimensionality reduction for tfidf
def reduce_dim_by_svd(X, ncomp):
    t0 = time()
    svd = TruncatedSVD(ncomp)
    X_res  = svd.fit_transform(X)
    print("done in %fs" % (time() - t0))
    explained_variance = svd.explained_variance_ratio_.sum()
    print("Explained variance of the SVD step: {}%".format(
        int(explained_variance * 100)))
    print()
    return X_res 
from gensim import matutils
Xtrvec = matutils.corpus2csc(Xtraintfidf).T.toarray()
Xtestvec = matutils.corpus2csc(Xtesttfidf).T.toarray()
X_tr_tfidf = reduce_dim_by_svd(Xtrvec, 100)
X_tst_tfidf = reduce_dim_by_svd(Xtestvec, 100)
from sklearn.model_selection import RandomizedSearchCV
from pprint import pprint
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 20, stop = 200, num = 5)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 50, num = 5)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
pprint(random_grid)
from sklearn.ensemble import RandomForestRegressor
# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 50 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 50, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(X_tr, y_train)
rf_random.best_params_
pd.DataFrame(rf_random.cv_results_).sort_values(by='mean_test_score', ascending=False).iloc[21]['params']
from sklearn.model_selection import GridSearchCV
# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}
# Create a base model
rf = RandomForestRegressor()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)
from sklearn.ensemble import RandomForestRegressor
regr = RandomForestRegressor(n_estimators=100, max_features='auto')
regr.fit(X_tr_tfidf,y_train)
print("R2 for training data: %0.2f\n" % regr.score(X_tr_tfidf,y_train))
print("R2 for test data: %0.2f\n" % regr.score(X_tst_tfidf,y_test))
y_pred = regr.predict(X_tst_tfidf)
rmse(y_pred,y_test)
adj_r2(y_pred,y_test,X_tst_tfidf.shape[1],len(y_pred)-1)
best_random = rf_random.best_estimator_
print("R2 for training data: %0.2f\n" % best_random.score(X_tr,y_train))
print("R2 for test data: %0.2f\n" % best_random.score(X_tst,y_test))
def adj_rsquare_rf(y_pred,y_act,npred):
    assert(len(y_pred)==len(y_act))
    resid = y_pred-y_act
    ssresid = np.sum(np.square(resid))
    sstot = np.sum(np.square(y_act-np.mean(y_act)))
    dftot = len(y_act)-1
    dfest = len(y_act)-npred-1
    r2adj = 1-ssresid*dftot/sstot/dfest
    return r2adj
pd.DataFrame(rf_random.cv_results_).sort_values(by='mean_test_score', ascending=False)
y_fit = best_random.predict(X_tr)
adj_rsquare_rf(y_fit,y_train,X_tr.shape[1])
adj_rsquare_rf(y_pred,y_test,X_tst.shape[1])
y_pred = best_random.predict(X_tst)
print(rmse(y_pred,y_test))
print(mape(y_pred,y_test))
