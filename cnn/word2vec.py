from corpusreader import TweetCorpusReader
from datetime import datetime
from gensim.models import word2vec
from random import choice
from string import hexdigits

import hyper_params, logging, os, sys, time
import ujson as json
import argparse

ON = 1

def get_model_name():
    rand_preffix = ''.join(choice(hexdigits) for i in xrange(6)).lower()
    ts = time.asctime().replace(" ", "_")
    return rand_preffix + '_' + ts + ".model"

def train_w2v_model(emb_dim, corpus_path, model_path):
    assert word2vec.FAST_VERSION is ON, "word2vec.FAST_VERSION is OFF"
    model_name = get_model_name()
    outpath = '{0}/{1}'.format(model_path, model_name)
    print " model will be stored on:     %s" % outpath
    print " metadata will be stored on:  %s" % "{0}.metadata".format(outpath); print

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    reader = TweetCorpusReader(corpus_path)
    model = word2vec.Word2Vec(reader, workers=4, window=3, size=emb_dim, min_count=3)
    model.init_sims(replace=True)

    if not os.path.exists(model_path):
        os.mkdir(model_path)

    model.save(outpath)
    with open("{0}.metadata".format(outpath), "w") as f:
        f.write("emb_dim=%d\n" % emb_dim)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--emb_dim', action="store", dest="emb_dim", type=int, default=0)
    parser.add_argument('--corpus_path', action="store", dest="corpus_path", default='../data')
    parser.add_argument('--model_path', action="store", dest="model_path", default='../models')

    args = parser.parse_args(); print
    print 'args.emb_dim     = ', args.emb_dim
    print 'args.corpus_path = ', args.corpus_path
    print 'args.model_path  = ', args.model_path; print

    if args.emb_dim == 0:
        for emb_dim in [20, 30, 50, 100, 200]:
            print "training model with emb_dim=%d" % emb_dim
            train_w2v_model(emb_dim, args.corpus_path, args.model_path)
    else:
        train_w2v_model(args.emb_dim, args.corpus_path, args.model_path)
