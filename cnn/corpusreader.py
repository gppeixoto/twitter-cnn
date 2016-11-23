import ujson as json
import hyper_params, logging, os, sys, time
from processor import process

TWEET_SUFFIX = "tweets.json"
TEXT = "text"
LABEL = "label"

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
