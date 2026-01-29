import os
import time
import re
import gensim
from multiprocessing import Pool
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
from gensim.models.phrases import ENGLISH_CONNECTOR_WORDS

def download_nltk_resources():
    import nltk
    for res in ['punkt', 'averaged_perceptron_tagger', 'wordnet', 'stopwords', 'omw-1.4']:
        nltk.download(res, quiet=True)

def tag2wordnettag(treebank_tag):
    if treebank_tag.startswith('J'): return wordnet.ADJ
    elif treebank_tag.startswith('V'): return wordnet.VERB
    elif treebank_tag.startswith('N'): return wordnet.NOUN
    elif treebank_tag.startswith('R'): return wordnet.ADV
    else: return None