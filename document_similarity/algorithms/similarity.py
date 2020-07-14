import math
import os
import re
import string

import nltk
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances

# removing stopwords from text
from nlp import model


def stopwords(text):
    # for adding new stopwords in ntlk-stopwords
    file = open(f"{os.path.dirname(__file__)}/turkce-stop-words.txt")
    stop_words_extra = file.read()
    stopWords = nltk.corpus.stopwords.words('turkish')
    for i in stop_words_extra.split():
        stopWords.append(i)

    str = ""
    words = [word for word in text.split() if word not in stopWords]
    for i in words:
        str += i
        str += " "
    return str


# converts capital letters to lowercase in text
def text_lowercase(text):
    return text.lower()


# removing numbers from text
def remove_numbers(text):
    result = re.sub(r'\d+', '', text)
    return result


# removing punctuations in text
def remove_punctuations(text):
    translator = str.maketrans("", "", string.punctuation)
    return text.translate(translator)


# removing whitespace in text
def remove_whitespace(text):
    return " ".join(text.split())


# tokenize text
def tokenizer(text):
    text = re.sub('^^\x20-\x7E', '', text)
    return word_tokenize(text)


# preprocessing for texts
def preprocessing(cleaned):
    cleaned = remove_whitespace(cleaned)
    cleaned = remove_numbers(cleaned)
    cleaned = remove_punctuations(cleaned)
    valid_characters = 'abcçdefgğhıijklmnoöpqrsştuüvwxyzQWERTYUIOPĞÜASDFGHJKLŞİZXCVBNMÖÇ1234567890 '
    cleaned = ''.join([x for x in cleaned if x in valid_characters])
    cleaned = text_lowercase(cleaned)
    cleaned = stopwords(cleaned)
    cleaned = tokenizer(cleaned)
    return cleaned


# preprocessing for texts
def preprocessing_(doc):
    cleaned = doc
    cleaned = remove_whitespace(cleaned)
    cleaned = remove_numbers(cleaned)
    cleaned = remove_punctuations(cleaned)
    cleaned = text_lowercase(cleaned)
    cleaned = stopwords(cleaned)
    return cleaned


# preprocessing for all documents
def alldocclean(alldocuments):
    predocs = []
    for i in alldocuments:
        predoc = preprocessing(i)
        predocs.append(predoc)
    return predocs  # predocs contains all cleared texts


# preprocessing for all documents
def alldocclean_(alldocuments):
    predocs = []
    for i in alldocuments:
        predoc = preprocessing_(i)
        predocs.append(predoc)
    return predocs  # predocs contains all cleared texts


# totalwords() -> get the cleaned document list
# return -> words in all documents
def totalwords(allpredocs):
    total = []
    for i in range(len(allpredocs)):
        if i <= len(allpredocs):
            total = set(total).union(set(allpredocs[i]))
    return total


# createDict -> get the cleaned documents list and totalwords list
# return -> a dictionary that contains words and the number of times words are repeated in the text for all texts
def createDict(allpredocs, totalwords):
    wordDicts = []
    for allpredoc in allpredocs:
        wordDictdoc = dict.fromkeys(totalwords, 0)
        for word in allpredoc:
            wordDictdoc[word] += 1
        wordDicts.append(wordDictdoc)
    return wordDicts


# Compute Term Frequency
# word count in document from wordDict diveded by number of total words in document
def computeTF(wordDict, bow):
    tfDict = {}
    bowCount = len(bow)
    for word, count in wordDict.items():
        tfDict[word] = count / float(bowCount)
    return tfDict


# Compute IDF for every word
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


# Compute TFIDF
# TF*IDF
def computeTFIDF(tfBow, idfs):
    tfidf = {}
    for word, val in tfBow.items():
        tfidf[word] = val * idfs[word]
    return tfidf


# Compute TF-IDF for all document
def TFIDF(alldocuments):
    predocs = alldocclean(alldocuments)  # clean all doc
    total = totalwords(predocs)  # list of words in all doc
    dicts = createDict(predocs, total)  # the number of times words repeat through the text
    tfs = []  # for values of tf in every doc
    for i in range(len(dicts)):
        tf = computeTF(dicts[i], predocs[i])  # compute tf for one document
        tfs.append(tf)  # adding tfs
    idf = computeIDF(dicts)  # compute idf for every word
    tfidfs = []  # for values of tf-idf in every doc
    for i in range(len(tfs)):
        tfidf = computeTFIDF(tfs[i], idf)  # compute tf-idf for one document
        tfidfs.append(tfidf)  # adding tfidfs list
    return tfidfs


def cosine_similarity_(vector1, vector2):
    dot_product = sum(p * q for p, q in zip(vector1, vector2))
    magnitude = math.sqrt(sum([val ** 2 for val in vector1])) * math.sqrt(sum([val ** 2 for val in vector2]))
    if not magnitude:
        return 0
    return dot_product / magnitude


def jaccard_similarity(vec1, vec2):
    intersection = set(vec1).intersection(set(vec2))
    union = set(vec1).union(set(vec2))
    return len(intersection) / len(union)


def TFIDFCosineSimilarity(doc, alldoc):
    cleantext = alldocclean_(alldoc)
    tfidf = TfidfVectorizer()
    tfs = tfidf.fit_transform(cleantext)
    df = pd.DataFrame(tfs.toarray(), columns=tfidf.get_feature_names())
    cossim = cosine_similarity(df)
    cossimdf = pd.DataFrame(cossim)
    cossimpair = []
    for index in range(len(alldoc)):
        if (doc != index):
            temp = []
            temp.append(index)
            temp.append(cossimdf.at[index, doc])
            cossimpair.append(temp)
    return cossimpair


def TFIDFEuclideanDistance(doc, alldoc):
    cleantext = alldocclean_(alldoc)
    tfidf = TfidfVectorizer()
    tfs = tfidf.fit_transform(cleantext)
    df = pd.DataFrame(tfs.toarray(), columns=tfidf.get_feature_names())
    eucsim = euclidean_distances(df)
    eucsimdf = pd.DataFrame(eucsim)
    eucsimpair = []
    for index in range(len(alldoc)):
        if (doc != index):
            temp = []
            temp.append(index)
            temp.append(eucsimdf.at[index, doc])
            eucsimpair.append(temp)
    return eucsimpair


def TFIDFManhattanDistance(doc, alldoc):
    cleantext = alldocclean_(alldoc)
    tfidf = TfidfVectorizer()
    tfs = tfidf.fit_transform(cleantext)
    df = pd.DataFrame(tfs.toarray(), columns=tfidf.get_feature_names())
    mansim = manhattan_distances(df)
    mansimdf = pd.DataFrame(mansim)
    mansimpair = []
    for index in range(len(alldoc)):
        if (doc != index):
            temp = []
            temp.append(index)
            temp.append(mansimdf.at[index, doc])
            mansimpair.append(temp)
    return mansimpair


def document_vector(doc):
    doc = [word for word in doc if word in model.vocab]
    return np.mean(model[doc], axis=0)


def word2VecCosineSimilarity(doc, alldoc):
    arr = alldocclean(alldoc)
    simArr = []
    doc_text = document_vector(arr[doc])
    for i in range(len(arr)):
        if doc != i:
            temp = [i, cosine_similarity_(doc_text, document_vector(arr[i]))]
            simArr.append(temp)
    return simArr


def word2VecEuclideanDistance(doc, alldoc):
    arr = alldocclean(alldoc)
    simArr = []
    doc_text = document_vector(arr[doc])
    for i in range(len(arr)):
        if doc != i:
            temp = [i, Euclidean(doc_text, document_vector(arr[i]))]
            simArr.append(temp)
    return simArr


def word2VecManhattanDistance(doc, alldoc):
    arr = alldocclean(alldoc)
    simArr = []
    doc_text = document_vector(arr[doc])
    for i in range(len(arr)):
        if doc != i:
            temp = [i, manhattan_distance(doc_text, document_vector(arr[i]))]
            simArr.append(temp)
    return simArr


def Euclidean(vec1, vec2):
    return math.sqrt(sum(math.pow((v1 - v2), 2) for v1, v2 in zip(vec1, vec2)))


def manhattan_distance(a, b):
    return sum(abs(e1 - e2) for e1, e2 in zip(a, b))
