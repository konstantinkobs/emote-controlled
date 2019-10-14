import gensim
from pathlib import Path
import csv
import datetime
import os
from pprint import pprint

import argparse

parser = argparse.ArgumentParser(description="Querying the Twitch embedding")
parser.add_argument("--model_dir", dest="model_dir", default="embedding/embedding", action="store", type=str, help="models main folder")
args = parser.parse_args()

model_dir = Path(args.model_dir)

print("loading embedding")
model = gensim.models.word2vec.Word2Vec.load(str(args.model_dir))
print("embedding loaded")

word_vectors = model.wv
del model

###############################################################################
# In the following there are two examples on how you can query the embedding. #
# Words need to be lowercased. Emotes not.                                    #
###############################################################################

#######################################
## Task 1: Detection of the odd word ##
#######################################

print("Detection of the odd word")
print("=========================")
print("youtube, twitch, instagram:")
pprint(word_vectors.doesnt_match("youtube twitch instagram".split(" ")))
print("\n")

###############################################
## Task 2: Words that fit in a given context ##
###############################################

print("Words that fit in a given context")
print("=================================")
print("monday, tuesday, wednesday:")
pprint(word_vectors.most_similar_cosmul("monday tuesday wednesday".split(" ")))
print("\n")

############################
## Task 3: Word relations ##
############################

print("Word relations")
print("==============")
print("Man relates to Woman as King to ...:")
pprint(word_vectors.most_similar_cosmul(positive=['king', 'woman'], negative=['man']))
print("\n")

###################################
## Task 4: Emote intensification ##
###################################

print("Emote intensifications")
print("======================")
print("LUL relates to OMEGALUL as FeelsGoodMan to ...:")
pprint(word_vectors.most_similar_cosmul(positive=['OMEGALUL', 'FeelsGoodMan'], negative=['LUL']))
