from pomegranate import *
from nltk.tokenize import word_tokenize
from nltk import download
from urllib import request
import string
import numpy as np


# url = "http://www.gutenberg.org/files/2600/2600-0.txt"
# response = request.urlopen(url)
# raw = response.read().decode('utf8')
# raw=raw.lower()
# raw=raw.encode("ascii",errors="backslashreplace")

# tokens = word_tokenize(raw)
# words = [word.lower() for word in tokens if word.isalpha()]
# text=str()
# for word in words:
# 	text+=word+' '

with open("wnp.txt", "r") as f:
	text = f.read()




init = np.array([0.51316,0.48684])

tr = np.array([[0.47468, 0.52532],
                             [0.51656, 0.48344]]) 
emissionprob = np.ones((2,27))
emissionprob*=1/27

chars=list(string.ascii_lowercase)
chars.append(' ')
states={}
for char in chars:
	states[char]=1/27

sequences=list(text)
model = HiddenMarkovModel("war and peace")


model.bake()
model.fit(sequences, labels=sequences)


# model.startprob_ = init
# model.transmat_ = trans_mat
# model.emissionprob_ = emissionprob


