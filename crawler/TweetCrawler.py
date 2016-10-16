# coding=utf-8

import sys
import gzip
import twitter
import ujson as json
import cPickle as pickle

class TweetCrawler:
    LANG = "lang"
    TEXT = "text"
    ENGLISH = "en"
    SARCASM = "#sarcasm"

    def __init__(self, key_file_path, keywords=None):
        self.key_file_path = key_file_path
        self.keywords = keywords
        self.stream = None

    def __is_not_sarcasm__(self, text):
        return text.find(TweetCrawler.SARCASM) == -1

    def __is_tweet_valid__(self, tweet):
        if tweet.has_key(TweetCrawler.LANG) and tweet.has_key(TweetCrawler.TEXT):
            lang = tweet[TweetCrawler.LANG]
            text = tweet[TweetCrawler.TEXT]
            return lang == TweetCrawler.ENGLISH and \
                self.__is_not_sarcasm__(text)

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

    def crawl_tweets(self, keywords=None):
        if keywords is not None:
            self.keywords = keywords
        self.init_stream()
        for tweet in self.stream:
            if self.__is_tweet_valid__(tweet):
                print tweet
                break
