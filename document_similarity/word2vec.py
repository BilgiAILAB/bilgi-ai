import math
import re
import string
import os
import numpy as np
from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity

from document_similarity.similarity_algorithms import stopWords

model = KeyedVectors.load_word2vec_format(f"{os.path.dirname(__file__)}/trmodel", binary=True)


def document_vector(doc):
    doc = [word for word in doc if word in model.vocab]
    return np.mean(model[doc], axis=0)


def stopwords(text):
    str = ""
    words = [word for word in text.split() if word not in stopWords]
    for i in words:
        str += i
        str += " "
    return str


def text_lowercase(text):
    return text.lower()


def remove_numbers(text):
    result = re.sub(r'\d+', '', text)
    return result


def remove_punctuations(text):
    translator = str.maketrans("", "", string.punctuation)
    return text.translate(translator)


def remove_whitespace(text):
    return " ".join(text.split())


def tokenizer(text):
    text = re.sub('^^\x20-\x7E', '', text)
    return word_tokenize(text)


def preprocessing(doc):
    cleaned = doc
    cleaned = remove_whitespace(cleaned)
    cleaned = remove_numbers(cleaned)
    cleaned = remove_punctuations(cleaned)
    cleaned = text_lowercase(cleaned)
    cleaned = stopwords(cleaned)
    cleaned = tokenizer(cleaned)
    return cleaned


def alldocclean(alldocuments):
    predocs = []
    for i in alldocuments:
        predoc = preprocessing(i)
        predocs.append(predoc)
    return predocs


def cosine_similarity(vector1, vector2):
    dot_product = sum(p * q for p, q in zip(vector1, vector2))
    magnitude = math.sqrt(sum([val ** 2 for val in vector1])) * math.sqrt(sum([val ** 2 for val in vector2]))
    if not magnitude:
        return 0
    return dot_product / magnitude


def documentsCosineSimilarity_v2(doc, alldoc):
    arr = alldocclean(alldoc)
    simArr = []
    for i in range(len(arr)):
        temp = []
        if (doc != i):
            temp.append(i)
            temp.append(cosine_similarity(document_vector(arr[doc]), document_vector(arr[i])))
            simArr.append(temp)
    print(simArr, "DENEME")
    return simArr


def Euclidean(vec1, vec2):
    return math.sqrt(sum(math.pow((v1 - v2), 2) for v1, v2 in zip(vec1, vec2)))


def documentsEuclideanDistance_v2(doc, alldoc):
    arr = alldocclean(alldoc)
    simArr = []
    for i in range(len(arr)):
        temp = []
        if (doc != i):
            temp.append(i)
            temp.append(Euclidean(document_vector(arr[doc]), document_vector(arr[i])))
            simArr.append(temp)

    return simArr
