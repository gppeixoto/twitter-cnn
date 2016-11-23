import ujson as json
import hyper_params, logging, os, sys, time
from tweet_parser import parse
from keras.preprocessing import sequence
from hyper_params import SEQUENCE_LENGTH
import numpy as np

TWEET_SUFFIX = "tweets.json"
TEXT = "text"
LABEL = "label"
PAD = "<PAD/>"

class TweetCorpusReader(object):
    """
    Class for reading efficiently the Twitter corpora.
    Reads and parses tweets on the fly, returning
        only those that have at least two tokens.
    """
    def __init__(self, data_path, text_only=True, w2i=None):
        super(TweetCorpusReader, self).__init__()
        self.data_path = data_path
        self.text_only = text_only
        self.embedding = False
        self.w2i = w2i
        self.unk = None if w2i is None else len(self.w2i)

    def load_and_process(self, doc):
        doc = json.loads(doc)
        text = parse(doc[TEXT])
        label = doc[LABEL]
        return (text, label)

    def __pad__(self, sequence):
        sequence = [token for token in sequence if token in self.w2i]
        len_seq = len(sequence)
        if len_seq > SEQUENCE_LENGTH:
            return sequence[-SEQUENCE_LENGTH:]
        padded = [PAD] * SEQUENCE_LENGTH
        padded[-len_seq:] = sequence
        padded = [self.w2i[token] for token in padded]

        return padded

    def __iter__(self):
        json_files = []
        for json_file in os.listdir(self.data_path):
            if json_file.endswith(TWEET_SUFFIX):
                json_files.append("{0}/{1}".format(self.data_path, json_file))
        for json_file in json_files:
            with open(json_file, "r") as f_in:
                tweets_in_file = [self.load_and_process(doc) for doc in f_in]
                if self.w2i is None:
                    for tweet, label in tweets_in_file:
                        if len(tweet) > 1:
                            yield tweet if self.text_only else (tweet, label)
                else:
                    for tweet, label in tweets_in_file:
                        if len(tweet) > 1:
                            indexes = self.__pad__(tweet)
                            yield (np.array(indexes), label)
