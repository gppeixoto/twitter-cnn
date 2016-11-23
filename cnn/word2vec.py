from corpusreader import TweetCorpusReader
from datetime import datetime
from gensim.models import word2vec
from processor import process
from random import choice
from string import hexdigits

import hyper_params, logging, os, sys, time
import ujson as json

def get_model_name():
    rand_preffix = ''.join(choice(hexdigits) for i in xrange(6)).lower()
    now = time.time()
    dt = datetime.fromtimestamp(now)
    ts = "%d%d%dT%d%d" % (dt.year, dt.month, dt.day, dt.hour, dt.minute)
    return rand_preffix + '_' + ts + ".model"

def train_w2v_model(corpus_path, model_path):
    assert word2vec.FAST_VERSION == 1
    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s',
        level=logging.INFO)

    model_name = get_model_name()
    corpus = TweetCorpusReader(corpus_path)
    model = word2vec.Word2Vec(
        corpus,
        workers=4,
        window=3,
        size=hyper_params.EMBEDDING_DIM)
    model.init_sims(replace=True)

    if not os.path.exists(model_path):
        os.mkdir(model_path)

    model.save('{0}/{1}'.format(model_path, model_name))

if __name__ == '__main__':
    corpus_path = sys.argv[-2]
    model_path = sys.argv[-1]
    train_w2v_model(corpus_path, model_path)
