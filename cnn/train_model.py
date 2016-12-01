import convnet
import numpy as np
import time
from tweet_parser import tokenizeRawTweetText
from corpusreader import TweetCorpusReader
from hyper_params import SEQUENCE_LENGTH, EMBEDDING_DIM, FILTER_LENGTHS, FEATURE_MAPS
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support as prfs
from time import time

model_path = '../models/2bda1f_Thu_Nov_24_10:07:34_2016.model'
cnn_100 = convnet.SingleLayerCNN(
    SEQUENCE_LENGTH, 100, [4, 5, 6, 7], FEATURE_MAPS, model_path)

split = .7
embedding_reader = TweetCorpusReader('../data/', w2i=cnn_100.word2index())
print 'Reading dataset for embedding...'
t = time()
x, y = map(lambda val: np.array(val), zip(*embedding_reader))
t = time() - t
print 'Finished reading in %.0fs; ' % t
# y = y.reshape(y.shape[0], 1)

# Step 1: get number of neg
nb_neg = y[:,1].sum()

# Step 2: get indices of positive samples
indices = np.arange(y.shape[0])
pos_mask = (y[:,0]==1).reshape(indices.shape[0],)
pos_indices = indices[pos_mask][:nb_neg]
neg_indices = indices[np.invert(pos_mask)]

assert pos_indices.shape[0] == neg_indices.shape[0]
assert any(pos_indices == neg_indices) is False

indices = np.concatenate([pos_indices, neg_indices])
np.random.shuffle(indices)
x_balanced = x[indices]
y_balanced = y[indices]
cut = int(split * x_balanced.shape[0])

x_train, y_train = (x_balanced[:cut], y_balanced[:cut])
x_test, y_test = (x_balanced[cut:], y_balanced[cut:])

###########################################################
################# ConvNet Evaluation ######################
###########################################################

print "\n fitting cnn model on %d samples..." % x_train.shape[0]
# hist = cnn_100.model.fit(x_train, y_train, nb_epoch=10)
# cnn_100.save_model(hist)
# print 'saved hist: ', str(hist)
cnn_200_pred = cnn_200.model.predict(x_test)

y_true = np.array([np.argmax(i) for i in y_test])
y_pred = np.array([np.argmax(i) for i in cnn_200_pred])

cnn_acc = accuracy_score(y_true, y_pred)
p, r, f, s = prfs(y_true, y_pred, average='macro')

print
print "cnn_20 stats: precision=%.3f, recall=%.3f, f-score=%.3f, acc: %.3f" % (p, r, f, cnn_acc)

###########################################################
################# LogReg/SVM Preparation ##################
###########################################################

print 'reading text data...'
t = time()
text_reader = TweetCorpusReader('../data/', text_only=False)
t = time() - t
print 'time elapsed: %.0fs' % t

x_text, y_text = zip(*[(' '.join(doc), label) for doc, label in text_reader])
vect = TfidfVectorizer(ngram_range=(1,2), tokenizer=tokenizeRawTweetText)
x_text = vect.fit_transform(x_text)
y_text = np.array(y_text)

nb_neg = (y_text == -1).sum()

# Step 2: get indices of positive samples
indices = np.arange(y_text.shape[0])
pos_mask = (y_text==1).reshape(indices.shape[0],)
pos_indices = indices[pos_mask][:nb_neg]
neg_indices = indices[np.invert(pos_mask)]

assert pos_indices.shape[0] == neg_indices.shape[0]
assert any(pos_indices == neg_indices) is False

indices = np.concatenate([pos_indices, neg_indices])

indices = np.sort(indices)
x_balanced = x_text[indices]
y_balanced = y_text[indices]

assert y_balanced.sum() == 0

###########################################################
################# LogReg/SVM Evaluation ###################
###########################################################
svm = LinearSVC()
lrg = LogisticRegression()

svm_stats = []
lrg_stats = []
for i, (tr, ts) in enumerate(KFold(n=y_balanced.shape[0], n_folds=3, shuffle=True)):
    print "Current fold: %d" % i
    x_train = x_balanced[tr]
    y_train = y_balanced[tr]
    x_test = x_balanced[ts]
    y_true = y_balanced[ts]

    print "Fitting svm..."
    t = time()
    svm.fit(x_train, y_train)
    t = time() - t
    print "Fitted svm in %.0fs" % t
    y_pred = svm.predict(x_test)
    acc = accuracy_score(y_pred=y_pred, y_true=y_true)
    p, r, f, s = prfs(y_pred=y_pred, y_true=y_true, average='macro')
    print "svm stats: accuracy=%.3f, precision=%.3f, recall=%.3f, f-score=%.3f" % (acc, p, r, f)
    print
    svm_stats.append((acc, p, r, f, t))

    print "Fitting lrg..."
    t = time()
    lrg.fit(x_train, y_train)
    t = time() - t
    print "Fitted lrg in %.0fs" % t
    y_pred = lrg.predict(x_test)
    acc = accuracy_score(y_pred=y_pred, y_true=y_true)
    p, r, f, s = prfs(y_pred=y_pred, y_true=y_true, average='macro')
    print "lrg stats: accuracy=%.3f, precision=%.3f, recall=%.3f, f-score=%.3f" % (acc, p, r, f)
    print
    lrg_stats.append((acc, p, r, f, t))
