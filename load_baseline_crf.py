"""
Loads the saved baseline model and asks for user prompts to predict. 

'train_baseline_crf.py' should be run first.
"""

import pickle
import pandas as pd
from utils_baseline import *
import nltk
from nltk.tokenize import WordPunctTokenizer
import sklearn_crfsuite
saved_model_dir = "baseline/saved_models/baseline_crf.sav"
try:
    crf = pickle.load(open(saved_model_dir, 'rb'))
except FileNotFoundError:
    print("No model .sav file found. Did you run train_baseline_crf.py first?")


def prepare_prompt(text_df):
    getter_prompt = SentenceGetter(text_df)
    sentences_prompt = getter_prompt.sentences
    return [sent2features(s) for s in sentences_prompt]

# while True:
#     print()
#     text = input("Enter text to be parsed (or 'quit' to cancel): ")
#     if text in ['quit', 'exit', 'cancel', 'abort']:
#         break
#     X_test = prepare_prompt(text)
#     y_pred = crf.predict(X_test)

#     print()
#     print("Predicted classes:")
#     print()

#     tokenized_text = WordPunctTokenizer().tokenize(text)

#     pad_length = max(len(word) for word in tokenized_text)

#     for word_tag_pair in zip(tokenized_text, y_pred[0]):
#         print("{: <20} {: <20}".format(*word_tag_pair))

#     print()