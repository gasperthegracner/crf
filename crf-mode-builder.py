# invoke libraries
from bs4 import BeautifulSoup as bs
from bs4.element import Tag

import nltk
from nltk import word_tokenize, pos_tag
from sklearn.model_selection import train_test_split
import pycrfsuite
import os
import os.path

import numpy as np
from sklearn.metrics import classification_report
import utils


def get_labels(doc):
    return [label for (token, postag, label) in doc]


############################
# EXECUTION BUILDING MODEL #
############################

files_path = "./TaggedDocs/"

allxmlfiles = utils.append_annotations(files_path)
soup = bs(allxmlfiles, "html5lib")

#
# identify the tagged element
#
docs = []
sents = []

for d in soup.find_all("document"):
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
        sents = sents + tags
    docs.append(sents)  # appends all the individual documents into one list


#
# Generating features
#

data = []

for i, doc in enumerate(docs):
    tokens = [t for t, label in doc]
    tagged = nltk.pos_tag(tokens)
    data.append([(w, pos, label)
                 for (w, label), (word, pos) in zip(doc, tagged)])


X = [utils.extract_features(doc) for doc in data]
y = [get_labels(doc) for doc in data]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


trainer = pycrfsuite.Trainer(verbose=True)

# Submit training data to the trainer
for xseq, yseq in zip(X_train, y_train):
    trainer.append(xseq, yseq)

# Set the parameters of the model
trainer.set_params({
    # coefficient for L1 penalty
    'c1': 0.1,

    # coefficient for L2 penalty
    'c2': 0.01,

    # maximum number of iterations
    'max_iterations': 200,

    # whether to include transitions that
    # are possible, but not observed
    'feature.possible_transitions': True
})

# Provide a file name as a parameter to the train function, such that
# the model will be saved to the file when training is finished
trainer.train('crf.model')

############################
# TESTING BUILDING MODEL   #
############################

tagger = pycrfsuite.Tagger()
tagger.open('crf.model')
y_pred = [tagger.tag(xseq) for xseq in X_test]

# Create a mapping of labels to indices
labels = {"keyword": 1, "NA": 0}

# Convert the sequences of tags into a 1-dimensional array
predictions = np.array([labels[tag] for row in y_pred for tag in row])
truths = np.array([labels[tag] for row in y_test for tag in row])

print(classification_report(
    truths, predictions,
    target_names=["keyword", "NA"]))
