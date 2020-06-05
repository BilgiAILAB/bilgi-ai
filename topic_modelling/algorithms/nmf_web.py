import numpy as np
import plotly.graph_objects as go
from gensim.models import CoherenceModel
from sklearn.decomposition import NMF as sk_NMF
from sklearn.feature_extraction.text import TfidfVectorizer

from topic_modelling.algorithms import preprocess, topic_distance


def nmf_optimum_coherence(corpus, start, end, step):
    cleaned_data, data_tokens, id2word, corpus = preprocess.preprocess(corpus=corpus)
    topic_numbers = []
    coherence_values = []

    for num_topics in range(start, end + 1, step):
        vectorizer = TfidfVectorizer()
        A = vectorizer.fit_transform(cleaned_data)
        nmf_model = sk_NMF(n_components=num_topics, init='nndsvd')
        W = nmf_model.fit_transform(A)  # document topic distribution
        H = nmf_model.components_  # topic word distribution
        feature_names = vectorizer.get_feature_names()  # terms
        doc_number = len(W)
        topic_number = len(H)
        word_count = 20
        word_distributions = []
        for topic in range(topic_number):
            top_indices = np.argsort(H[topic, :])[::-1]
            doc_list = []
            for term_index in top_indices[0:word_count]:
                doc_list.append([feature_names[term_index], H[topic, term_index]])
            word_distributions.append(doc_list)
        nmf_topics = [[word[0] for word in topic] for topic in word_distributions]
        coherence = CoherenceModel(topics=nmf_topics, texts=data_tokens, dictionary=id2word).get_coherence()

        topic_numbers.append(num_topics)
        coherence_values.append(coherence)
    fig = go.Figure(data=go.Scatter(x=topic_numbers, y=coherence_values))
    return fig


def NMF(corpus, n_topic):
    cleaned_data, data_tokens, id2word, corpus = preprocess.preprocess(corpus=corpus)

    vectorizer = TfidfVectorizer()
    A = vectorizer.fit_transform(cleaned_data)
    nmf_model = sk_NMF(n_components=n_topic, init='nndsvd')
    W = nmf_model.fit_transform(A)  # document topic distribution
    H = nmf_model.components_  # topic word distribution
    feature_names = vectorizer.get_feature_names()  # terms
    doc_number = len(W)
    topic_number = len(H)
    word_count = 20

    word_distributions = []
    for topic in range(topic_number):
        top_indices = np.argsort(H[topic, :])[::-1]
        doc_list = []
        for term_index in top_indices[0:word_count]:
            doc_list.append([feature_names[term_index], H[topic, term_index]])
        word_distributions.append(doc_list)

    topic_distributions = []
    for document in range(doc_number):
        topic_distributions.append([[topic, W[document][topic]] for topic in range(len(W[document]))])

    doc_dist = {}
    for i in range(topic_number):
        doc_dist.update({i: []})

    for i in range(doc_number):
        doc_dist[topic_distance.get_topic_dist_max(topic_distributions[i])[0]].append(i)

    nmf_topics = [[word[0] for word in topic] for topic in word_distributions]
    coherence = CoherenceModel(topics=nmf_topics, texts=data_tokens, dictionary=id2word).get_coherence()

    output = {"filecount": doc_number,
              "coherence_value": float(coherence),
              "word_distributions": word_distributions,
              "topic_distributions": topic_distributions,
              "doc_dist": doc_dist,
              "data_tokens": data_tokens
              }

    return output

    # def my_converter(o):
    #     return o.__str__()
    #
    # with open('output_first.txt', 'w', encoding='utf-8') as outfile:
    #     json.dump(output, outfile, indent=4, default=my_converter, ensure_ascii=False)
    #
    # with open('output_first.txt', 'r', encoding="utf8") as handle:
    #     parsed = json.load(handle)
