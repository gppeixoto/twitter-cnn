# coding=utf-8
import sys
import codecs
import twitter
import ujson as json
import cPickle as pickle
import time
from datetime import datetime
from random import choice
from string import hexdigits

POS = [":)", "(:", ":-)", "(-:"]
NEG = [":(", "):", ":-(", ")-:"]

class TweetCrawler:
    LANG = "lang"
    TEXT = "text"
    ENGLISH = "en"
    NEW_LINE = "\n"

    def __init__(self, key_file_path, keywords=None, buffer_limit=100):
        self.key_file_path = key_file_path
        self.keywords = keywords
        self.stream = None
        self.buffer = []
        self.buffer_limit = buffer_limit

    def trim(self, text):
        text = text.replace("\n", ").strip()
        label = +1 if any(text.find(i) != -1 for i in POS) else -1
        return json.dumps({"text": text, "label": label})

    def __is_text_valid(self, text):
        return text != None and text != "" and len(text) > 5

    def __is_tweet_valid__(self, tweet):
        if tweet.has_key(TweetCrawler.LANG) and tweet.has_key(TweetCrawler.TEXT):
            lang = tweet[TweetCrawler.LANG]
            text = tweet[TweetCrawler.TEXT]
            return lang == TweetCrawler.ENGLISH and \
                self.__is_text_valid(text)

        return False

    def get_keys(self):
        with open(self.key_file_path, "r") as keyfile:
            keys = [key.strip() for key in keyfile]
        return keys

    def get_api_connection(self):
        keys = self.get_keys()
        return twitter.Api(keys[0], keys[1], keys[2], keys[3])

    def init_stream(self):
        api = self.get_api_connection()
        self.stream = api.GetStreamFilter(track=self.keywords)

    def __random_prefix__(self):
        return ".join([choice(hexdigits) for i in xrange(6)])

    def is_buffer_full(self):
        return len(self.buffer) >= self.buffer_limit

    def get_file_name(self):
        now = time.time()
        dt = datetime.fromtimestamp(now)
        ts = "%d%d%dT%d%d" % (dt.year, dt.month, dt.day, dt.hour, dt.minute)
        fname = self.__random_prefix__() + "_" + ts + "_tweets.json"
        return fname

    def dump_buffer(self, path):
        fname = path + self.get_file_name()
        with codecs.open(fname, "w", "utf-8") as f:
            for tweet in self.buffer:
                tweet = self.trim(tweet)
                f.write(tweet + TweetCrawler.NEW_LINE)
        self.buffer = []

    def crawl_tweets(self, keywords=None, path="./data/"):
        if keywords is not None:
            self.keywords = keywords
        self.init_stream()
        for i, tweet in enumerate(self.stream):
            print "Tweet: %d" % i
            if self.__is_tweet_valid__(tweet) and not self.is_buffer_full():
                self.buffer.append(tweet[TweetCrawler.TEXT])
                print "Valid on buffer: %d" % len(self.buffer)
            elif self.is_buffer_full():
                print "Dumping...\n...\n"
                self.dump_buffer(path)
            else:
                continue

def main():
    keywords = POS + NEG
    limit = sys.argv[-1]
    crawler = TweetCrawler("./crawler/keys.txt",
       keywords=keywords,
       buffer_limit=1000)
    crawler.crawl_tweets()

if __name__ == "__main__":
    main()
