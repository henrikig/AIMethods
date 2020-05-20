from SRNClassifier import SRNClassifier
import pickle
import gzip
import sys
import json
import numpy as np

with gzip.open("clf.data") as f:
    clf = pickle.load(f)

with open("dict.json") as f:
    word_dict = json.load(f)
    print(type(word_dict))


clf.reset()

priming_words = ["It", "was", "a"]
for word in priming_words:
    sys.stdout.write(word + " ")
    x = np.array(word_dict[word])
    yhat = clf.predict(x)

for i in range(200):
    # Extract word from one hot encoding
    for k, v in word_dict.items():
        if list(yhat) == v:
            if k not in ",.!?;":
                sys.stdout.write(" ")
            sys.stdout.write(k)
            break
    yhat = clf.predict(yhat)

while True:
    for k, v in word_dict.items():
        if list(yhat) == v:
            if k not in ",.!?;":
                sys.stdout.write(" ")
            sys.stdout.write(k)
            word = k
            break
    if word == ".":
        break
    yhat = clf.predict(yhat)

sys.stdout.flush()