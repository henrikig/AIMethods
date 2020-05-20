from SRNClassifier import SRNClassifier
import pickle
import gzip
import numpy as np
import re
from sklearn import preprocessing
import json


FILENAME = "corpus/corpus.txt"

# Extract list of words from corpus
with open(FILENAME, "r") as f:
    text = f.read()
    text = re.findall(r"[\w’]+|[.,!?;]", text)

# Remove duplicate words
words = list(set(text))

# Create one hot encoding from list of words
y = np.expand_dims(words, axis=1)
enc = preprocessing.OneHotEncoder()
enc.fit(y)
y = enc.transform(y).toarray()

# Map word to one hot encoding in dictionary
word_dict = {words[i]: list(y[i]) for i in range(len(words))}

# Initiate recurrent neural network classifier
clf = SRNClassifier(alpha=1e-2, hidden_layer_size=32, activation="tanh", max_iter=300, verbose=True, target=0.96)

x = np.array([np.array(word_dict[word]) for word in text][:-1])
pred = np.array([np.array(word_dict[word]) for word in text][1:])


clf.fit(x, pred)

with gzip.open("clf.data", "w") as f:
    pickle.dump(clf, f)

with open("dict.json", "w") as f:
    json.dump(word_dict, f)
