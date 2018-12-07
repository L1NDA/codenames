import numpy as np
import random
import gensim
from nltk.corpus import words
from functools import reduce
import heapq

wrds = np.genfromtxt('wordlist.csv', delimiter=',', dtype=str).tolist()

board = [x[random.random() > 0.5].lower() for x in random.sample(wrds, 25)]

player1 = board[:9]
player2 = board[9:17]
neutral = board[17:24]
assassin = board[24]

model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
