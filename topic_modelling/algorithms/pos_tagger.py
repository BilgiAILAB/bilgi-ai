# -*- coding: cp1254 -*-

# Onur Yilmaz

# Imports

# Open the file where tagger is saved
from topic_modelling.algorithms.Tagger import Tagger

taggerFileName = 'topic_modelling/algorithms/my_tagger.yaml'
myTagger = Tagger.load(taggerFileName)

# Keep the original functionality intact
def tag(sentence):
    return myTagger.tag(sentence)

# End of code
