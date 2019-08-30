from bs4 import BeautifulSoup as bs
from bs4.element import Tag
import codecs
import nltk
from nltk import word_tokenize, pos_tag
from sklearn.model_selection import train_test_split
import pycrfsuite
import os
import os.path
import sys
import glob
from xml.etree import ElementTree
import numpy as np
from sklearn.metrics import classification_report


def remov_punct(withpunct):
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    without_punct = ""
    char = 'nan'
    for char in withpunct:
        if char not in punctuations:
            without_punct = without_punct + char
    return(without_punct)

# this function appends all annotated files


def append_annotations(files):
    xml_files = glob.glob(files + "/*.xml")
    xml_element_tree = None
    new_data = ""
    for xml_file in xml_files:
        data = ElementTree.parse(xml_file).getroot()
        # print ElementTree.tostring(data)
        temp = ElementTree.tostring(data)
        new_data += (temp)
    return(new_data)


def word2features(doc, i):
    word = doc[i][0]
    postag = doc[i][1]

    # Common features for all words. You may add more features here based on your custom use case
    features = [
        'bias',
        'word.lower=' + word.lower(),
        'word[-3:]=' + word[-3:],
        'word[-2:]=' + word[-2:],
        'word.isupper=%s' % word.isupper(),
        'word.istitle=%s' % word.istitle(),
        'word.isdigit=%s' % word.isdigit(),
        'postag=' + postag
    ]

    # Features for words that are not at the beginning of a document
    if i > 0:
        word1 = doc[i-1][0]
        postag1 = doc[i-1][1]
        features.extend([
            '-1:word.lower=' + word1.lower(),
            '-1:word.istitle=%s' % word1.istitle(),
            '-1:word.isupper=%s' % word1.isupper(),
            '-1:word.isdigit=%s' % word1.isdigit(),
            '-1:postag=' + postag1
        ])
    else:
        # Indicate that it is the 'beginning of a document'
        features.append('BOS')

    # Features for words that are not at the end of a document
    if i < len(doc)-1:
        word1 = doc[i+1][0]
        postag1 = doc[i+1][1]
        features.extend([
            '+1:word.lower=' + word1.lower(),
            '+1:word.istitle=%s' % word1.istitle(),
            '+1:word.isupper=%s' % word1.isupper(),
            '+1:word.isdigit=%s' % word1.isdigit(),
            '+1:postag=' + postag1
        ])
    else:
            # Indicate that it is the 'end of a document'
        features.append('EOS')

    return features


def extract_features(doc):
    return [word2features(doc, i) for i in range(len(doc))]
