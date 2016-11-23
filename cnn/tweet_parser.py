import ujson as json
import re
from twokenize import tokenizeRawTweetText
from nltk.corpus import stopwords
from string import digits, punctuation
from hyper_params import SEQUENCE_LENGTH
import re

# "not" shouldn't be a stop word, given it's a polarity changer
EN_STOP = stopwords.words('english')
EN_STOP.remove('not')

urlrxp = re.compile(
        r'^(?:http|ftp)s?://' # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|' #domain...
        r'localhost|' #localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})' # ...or ip
        r'(?::\d+)?' # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE
        )

# Should remove the emojis from the text,
# so it won't bias the classifier.
POS = [':)', '(:', ':-)', '(-:']
NEG = [':(', '):', ':-(', ')-:']
EMOJIS = POS + NEG
BAD_CHARS = digits + punctuation
CUT = 3
MENTION = "<MNT/>"
URL = "<URL/>"
PAD = "<PAD/>"

def is_url(token):
    return urlrxp.match(token) is not None

def only_digits_or_punctuation(token):
    return all(i in BAD_CHARS for i in token)

def is_too_short(token):
    return len(token) < CUT

def pad(sequence):
    len_seq = len(sequence)
    if len_seq > SEQUENCE_LENGTH:
        return sequence[-SEQUENCE_LENGTH:]
    padded = [PAD] * SEQUENCE_LENGTH
    padded[-len_seq:] = sequence
    return padded

def should_be_kept(token):
    token = token.lower()
    return not (token in EN_STOP or \
        token in EMOJIS or \
        only_digits_or_punctuation(token) or \
        is_too_short(token))

def placeholder_and_lower(token):
    if token[0] == "@":
        return MENTION
    elif is_url(token):
        return URL
    else:
        return token.lower()

def parse(text):
    tokens = tokenizeRawTweetText(text)
    tokens = filter(should_be_kept, tokens)
    tokens = map(placeholder_and_lower, tokens)
    return pad(tokens)
