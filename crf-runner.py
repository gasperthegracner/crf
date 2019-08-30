# invoke libraries
from bs4 import BeautifulSoup as bs
import codecs
import nltk
from nltk import word_tokenize, pos_tag

import pycrfsuite

import utils

############################
# EVALUATION
############################

with codecs.open("./toPredict.xml", "r", "utf-8") as infile:
    soup_test = bs(infile, "html5lib")

docs = []
sents = []

for d in soup_test.find_all("document"):
    for wrd in d.contents:
        tags = []
        NoneType = type(None)

        if isinstance(wrd.name, NoneType) == True:
            withoutpunct = utils.remov_punct(wrd)
            temp = word_tokenize(withoutpunct)
            for token in temp:
                tags.append((token, 'NA'))
        else:
            withoutpunct = utils.remov_punct(wrd)
            temp = word_tokenize(withoutpunct)
            for token in temp:
                tags.append((token, wrd.name))
        # docs.append(tags)

sents = sents + tags  # puts all the sentences of a document in one element of the list
docs.append(sents)  # appends all the individual documents into one list

data_test = []

for i, doc in enumerate(docs):
    tokens = [t for t, label in doc]
    tagged = nltk.pos_tag(tokens)
    data_test.append([(w, pos, label)
                      for (w, label), (word, pos) in zip(doc, tagged)])

tagger = pycrfsuite.Tagger()


data_test_feats = [utils.extract_features(doc) for doc in data_test]
tagger.open('crf.model')
newdata_pred = [tagger.tag(xseq) for xseq in data_test_feats]

# Let's check predicted data
i = 0
for x, y in zip(newdata_pred[i], [x[1].split("=")[1] for x in data_test_feats[i]]):
    print("%s (%s)" % (y, x))
