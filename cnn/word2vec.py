from datetime import datetime
from gensim.models import word2vec
from processor import process
from random import choice
from string import hexdigits

import os
import time
import ujson as json

TWEET_SUFFIX = "tweets.json"
TEXT = "text"
LABEL = "label"
MODELS_DIR = "w2v_models"

class TweetCorpusReader(object):
    """
    Class for reading efficiently the Twitter corpora.
    Reads and parses tweets on the fly, returning
        only those that have at least two tokens.
    """
    def __init__(self, data_path, text_only=True):
        super(TweetCorpusReader, self).__init__()
        self.data_path = data_path
        self.text_only = text_only

    def load_and_process(self, doc):
        doc = json.loads(doc)
        text = process(doc[TEXT])
        label = doc[LABEL]
        return (text, label)

    def __iter__(self):
        json_files = []
        for json_file in os.listdir(self.data_path):
            if json_file.endswith(TWEET_SUFFIX):
                json_files.append("{0}/{1}".format(self.data_path, json_file))
        for json_file in json_files:
            with open(json_file, "r") as f_in:
                tweets_in_file = [self.load_and_process(doc) for doc in f_in]
                for tweet, label in tweets_in_file:
                    if len(tweet) > 1:
                        yield tweet if self.text_only else (tweet, label)

def get_model_name():
    rand_preffix = ''.join(choice(hexdigits) for i in xrange(6)).lower()
    now = time.time()
    dt = datetime.fromtimestamp(now)
    ts = "%d%d%dT%d%d" % (dt.year, dt.month, dt.day, dt.hour, dt.minute)
    return rand_preffix + '_' + ts + ".model"


def train_w2v_model(corpus_path, model_path):
    assert word2vec.FAST_VERSION == 1

    model_name = get_model_name()
    print "training model: {0}...".format(model_name)
    corpus = TweetCorpusReader(corpus_path)
    model = word2vec.Word2Vec(corpus, workers=4, window=3)

    model.init_sims(replace=True)
    if not os.path.exists(MODELS_DIR):
        os.mkdir(MODELS_DIR)

    model.save('{0}/{1}'.format(model_path, model_name))
