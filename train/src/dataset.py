import json
import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import random
import numpy as np
from pathlib import Path


class IntentLoader:
    def _load_intents(self, fp):
        with open(fp, 'r') as f:
            intents = json.load(f)
        return intents

    def __init__(self, fp):
        if type(fp) is str or isinstance(fp, Path):
            self.intents = self._load_intents(fp)
        elif type(fp) is dict:
            self.intents = fp
        else:
            raise Exception("Invalid data type.")

        self.classes = []
        self.documents = []
        self.words = []
        self.ignore_letters = ['!', '?', ',', '.']
        self.lemmatizer = WordNetLemmatizer()
        
    def fit(self):
        for intent in self.intents['intents']:
            for pattern in intent['patterns']:
                word = nltk.word_tokenize(pattern)
                self.words.extend(word)
                self.documents.append((word, intent['tag']))
                self.classes.append(intent['tag'])

        # Lemmatize, remove duplicates and sort
        self.words = [self.lemmatizer.lemmatize(w.lower()) for w in self.words if w not in self.ignore_letters]
        self.words = sorted(list(set(self.words)))
        # Sort classes
        self.classes = sorted(list(set(self.classes)))
        print(f"Fit {len(self.documents)} documents for {len(self.classes)} classes (total words: {len(self.words)})")

    def describe(self):
        if not self.classes:
            print("No class found. Perform IntentLoader.fit() first.")
            return
        print(f"N° of documents: {len(self.documents)}")
        print(f"N° of classes: {len(self.classes)}\n{self.classes}")
        print(f"N° of words: {len(self.words)}\nList of unique words:\n{self.words}")  

    def __str__(self):
        return f"N° of categories: {len(self.intents['intents'])}"

    def save_model(self, fp):
        with open(fp,'wb') as file:
            pickle.dump(il, file)

    def to_train_set(self):
        train_set = []
        output_empty = [0] * len(self.classes)

        for doc_words, tag in self.documents:
            bag = []

            # Preprocessing of the words to compare them to the list of words in the corpus
            word_patterns = [self.lemmatizer.lemmatize(word.lower()) for word in doc_words]

            # Go through all the words and signal if the document word is there or not
            for word in self.words:
                bag.append(1) if word in word_patterns else bag.append(0)

            # Marks the tag from the output
            output_row = output_empty[:]
            output_row[self.classes.index(tag)] = 1

            # First column is the bow data, second column is the class
            train_set.append([bag, output_row])


        random.shuffle(train_set)
        train_set = np.array(train_set, dtype=object)

        train_x, train_y = list(train_set[:,0]), list(train_set[:,1])

        print("Training data is created")
        print(f"N° samples X: {len(train_x)}, n° samples y {len(train_y)}")

        return train_x, train_y

