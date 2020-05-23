import math
import os
import re
import string

import nltk
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity

corpus = []
docname = []
c = []

file = open(f"{os.path.dirname(__file__)}/turkce-stop-words.txt")
text = file.read()
stopWords = nltk.corpus.stopwords.words('turkish')
for i in text.split():
    stopWords.append(i)


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


def totalwords(allpredocs):
    total = []
    for i in range(len(allpredocs)):
        if i <= len(allpredocs):
            total = set(total).union(set(allpredocs[i]))
    return total


def createDict(allpredocs, totalwords):
    wordDicts = []
    for allpredoc in allpredocs:
        wordDictdoc = dict.fromkeys(totalwords, 0)
        for word in allpredoc:
            wordDictdoc[word] += 1
        wordDicts.append(wordDictdoc)
    return wordDicts


def computeTF(wordDict, bow):
    tfDict = {}
    bowCount = len(bow)
    for word, count in wordDict.items():
        tfDict[word] = count / float(bowCount)
    return tfDict


def computeIDF(docList):
    idfDict = {}
    N = len(docList)

    idfDict = dict.fromkeys(docList[0].keys(), 0)
    for doc in docList:
        for word, val in doc.items():
            if val > 0:
                idfDict[word] += 1

    for word, val in idfDict.items():
        idfDict[word] = math.log10(N / float(val))

    return idfDict


def computeTFIDF(tfBow, idfs):
    tfidf = {}
    for word, val in tfBow.items():
        tfidf[word] = val * idfs[word]
    return tfidf


def TFIDF(alldocuments):
    predocs = alldocclean(alldocuments)
    total = totalwords(predocs)
    dicts = createDict(predocs, total)
    tfs = []
    for i in range(len(dicts)):
        tf = computeTF(dicts[i], predocs[i])
        tfs.append(tf)
    idf = computeIDF(dicts)
    tfidfs = []
    for i in range(len(tfs)):
        tfidf = computeTFIDF(tfs[i], idf)
        tfidfs.append(tfidf)
    return tfidfs


def cosine_similarity(vector1, vector2):
    dot_product = sum(p * q for p, q in zip(vector1, vector2))
    magnitude = math.sqrt(sum([val ** 2 for val in vector1])) * math.sqrt(sum([val ** 2 for val in vector2]))
    if not magnitude:
        return 0
    return dot_product / magnitude


def documentsCosineSimilarity(doc, alldoc):
    arr = TFIDF(alldoc)
    simArr = []
    for i in range(len(arr)):
        temp = []
        if doc != i:
            temp.append(i)
            temp.append(cosine_similarity(arr[doc].values(), arr[i].values()))
            simArr.append(temp)

    return simArr


def Euclidean(vec1, vec2):
    return math.sqrt(sum(math.pow((v1 - v2), 2) for v1, v2 in zip(vec1, vec2)))


def documentsEuclideanDistance(doc, alldoc):
    arr = TFIDF(alldoc)
    simArr = []
    for i in range(len(arr)):
        temp = []
        if (doc != i):
            temp.append(i)
            temp.append(Euclidean(arr[doc].values(), arr[i].values()))
            simArr.append(temp)
    return simArr


def jaccard_similarity(vec1, vec2):
    intersection = set(vec1).intersection(set(vec2))
    union = set(vec1).union(set(vec2))
    return len(intersection) / len(union)


def documentsJaccardSimilarity(doc, alldoc):
    arr = TFIDF(alldoc)
    simArr = []
    for i in range(len(arr)):
        temp = []
        if doc != i:
            temp.append(i)
            temp.append(jaccard_similarity(arr[doc].values(), arr[i].values()))
            simArr.append(temp)
    return simArr


def manhattan_distance(a, b):
    return sum(abs(e1 - e2) for e1, e2 in zip(a, b))


def documentsManhattanDistance(doc, alldoc):
    arr = TFIDF(alldoc)
    simArr = []
    for i in range(len(arr)):
        temp = []
        if doc != i:
            temp.append(i)
            temp.append(manhattan_distance(arr[doc].values(), arr[i].values()))
            simArr.append(temp)
    return simArr
