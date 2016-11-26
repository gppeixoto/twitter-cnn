import convnet
import numpy as np
import time
from tweet_parser import tokenizeRawTweetText
from corpusreader import TweetCorpusReader
from hyper_params import SEQUENCE_LENGTH, EMBEDDING_DIM, FILTER_LENGTHS, FEATURE_MAPS

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support as prfs
from time import time

# TODO: test passing generator to avoid storing data in memory

model_path = '../models/f40db9_Thu_Nov_24_10:38:27_2016.model' #emb_dim=200
cnn_200 = convnet.SingleLayerCNN(
    SEQUENCE_LENGTH, 200, [4, 5, 6, 7], FEATURE_MAPS, model_path)

split = .7
embedding_reader = TweetCorpusReader('../data/', w2i=cnn_200.word2index())
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

hist = cnn_200.model.fit(x_train, y_train, nb_epoch=10)
cnn_200.save_model()
cnn_pred = cnn_200.model.predict(x_test)

y_true = np.array([np.argmax(i) for i in y_test])
y_pred = np.array([np.argmax(i) for i in cnn_pred])

cnn_acc = accuracy_score(y_true, y_pred)
p, r, f, s = prfs(y_true, y_pred, average='macro')

print
print "cnn stats: accuracy=%.3f, precision=%.3f, recall=%.3f, f-score=%.3f" % (cnn_acc, p, r, f)

text_reader = TweetCorpusReader('../data/', text_only=False)

x_text, y_text = zip(*[(' '.join(doc), label) for doc, label in text_reader])
vect = TfidfVectorizer(ngram_range=(1,2), tokenizer=tokenizeRawTweetText)
x_text = vect.fit_transform(x_text)

svm = LinearSVC()
lrg = LogisticRegression()

svm.fit(x_text[:cut], y_text[:cut])
lrg.fit(x_text[:cut], y_text[:cut])

svm_pred = svm.predict(x_text[cut:])
lrg_pred = svm.predict(x_text[cut:])

svm_acc = accuracy_score(y_text[cut:], svm_pred)
lrg_acc = accuracy_score(y_text[cut:], lrg_pred)

p, r, f, s = prfs(y_text[cut:], svm_pred, average='macro')
print "svm stats: accuracy=%.3f, precision=%.3f, recall=%.3f, f-score=%.3f" % (svm_acc, p, r, f)

p, r, f, s = prfs(y_text[cut:], lrg_pred, average='macro')
print "lrg stats: accuracy=%.3f, precision=%.3f, recall=%.3f, f-score=%.3f" % (lrg_acc, p, r, f)
