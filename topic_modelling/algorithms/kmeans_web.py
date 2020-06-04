import numpy as np
import plotly.graph_objects as go
from gensim.models import LdaModel
from sklearn import metrics
from sklearn.cluster import KMeans

from document_similarity.algorithms import similarity
from nlp import model
from topic_modelling.algorithms import distributions, preprocess


def kmeans_optimum_value(corpus, start, end, step):
    doc_vectors, cleaned_data = doc_vector_generator(corpus)

    s_scores = []
    k_values = range(start, end, step)

    for i in k_values:
        i_scores = []
        for j in range(1):
            kmeans_model = KMeans(n_clusters=i, init='k-means++', n_init=40)
            kmeans_model.fit(doc_vectors)
            labels = kmeans_model.labels_.tolist()
            i_scores.append(metrics.silhouette_score(doc_vectors, labels, metric='cosine'))
        s_scores.append(np.mean(i_scores))

    fig = go.Figure(data=go.Scatter(x=[k for k in k_values], y=s_scores))
    fig.update_layout(title="Finding Optimum k",
                      xaxis_title='Cluster Numbers, k',
                      yaxis_title='Silhouette Score')
    return fig


def doc_vector_generator(corpus):
    def document_vector(doc):
        doc = [word for word in doc if word in model.vocab]
        return np.mean(model[doc], axis=0)

    cleaned_data = similarity.alldocclean(corpus)
    doc_vectors = [document_vector(c) for c in cleaned_data]

    return doc_vectors, cleaned_data


def w2v_kmeans(corpus, n_clusters):
    doc_vectors, cleaned_data_tokens = doc_vector_generator(corpus)

    kmeans_model = KMeans(n_clusters=n_clusters, init='k-means++', n_init=40)
    kmeans_model.fit(doc_vectors)
    labels = kmeans_model.labels_.tolist()
    doc_number = len(labels)

    nested_corpus = []
    for i in range(n_clusters):
        nested_corpus.append([])

    for i in range(doc_number):
        nested_corpus[labels[i]].append(corpus[i])

    doc_dist = {}
    document_dists = np.array(labels)
    for cluster in range(n_clusters):
        doc_dist.update({cluster: np.where(document_dists == cluster)[0].tolist()})

    topic_distributions = []
    for i in range(doc_number):
        topic_distributions.append([[labels[i], 1.0]])

    word_distributions = []
    for cluster_number in range(len(nested_corpus)):
        cleaned_data, data_tokens, id2word, corpus3 = preprocess.preprocess(corpus=nested_corpus[cluster_number])
        n_topic = 1
        lda_model = LdaModel(corpus=corpus3,
                             id2word=id2word,
                             num_topics=n_topic,
                             random_state=100,
                             update_every=1,
                             # chunksize=50,
                             passes=10,
                             alpha='auto',
                             per_word_topics=True,
                             minimum_probability=1e-8)
        word_distributions.append(distributions.word_distribution(model=lda_model, n_topic=n_topic)[0])
    silhouette_score = metrics.silhouette_score(doc_vectors, labels, metric='cosine')
    output = {"filecount": doc_number,
              "silhouette_score": float(silhouette_score),
              "word_distributions": word_distributions,
              "topic_distributions": topic_distributions,
              "doc_dist": doc_dist,
              "data_tokens": cleaned_data_tokens,
              "labels": labels,
              "doc_vectors": [doc_vec.tolist() for doc_vec in doc_vectors]
              }
    return output
