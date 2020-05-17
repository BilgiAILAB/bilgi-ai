import re

import gensim
import nltk
from gensim.models import CoherenceModel, LdaModel

import plotly.graph_objects as go
from topic_modelling import preprocess

def get_topic_dist_max(vector):
    dict_of_topics = dict(vector)
    maximum_topic = max(dict_of_topics, key=dict_of_topics.get)
    return maximum_topic, dict_of_topics.get(maximum_topic)


def LDA_optimum_coherence(corpus, id2word, data_tokens, start, end, step):
    topic_numbers = []
    coherence_values = []

    for num_topics in range(start=start, stop=end, step=step):
        lda = LdaModel(corpus=corpus,
                       id2word=id2word,
                       num_topics=num_topics,
                       random_state=100,
                       update_every=1,
                       chunksize=50,
                       passes=10,
                       alpha='auto',
                       per_word_topics=True,
                       minimum_probability=1e-8)
        coherence = CoherenceModel(model=lda, texts=data_tokens, dictionary=id2word, coherence='c_v').get_coherence()
        topic_numbers.append(num_topics)
        coherence_values.append(coherence)
    fig = go.Figure(data=go.Scatter(x=topic_numbers, y=coherence_values))
    return fig.show()


def LDA(corpus, n_topic):

    cleaned_data, data_tokens, id2word, corpus = preprocess.preprocess(corpus=corpus)


    lda_model = LdaModel(corpus=corpus,
                         id2word=id2word,
                         num_topics=n_topic,
                         random_state=100,
                         update_every=1,
                         chunksize=50,
                         passes=10,
                         alpha='auto',
                         per_word_topics=True,
                         minimum_probability=1e-8)

    cm = CoherenceModel(model=lda_model, texts=data_tokens, dictionary=id2word, coherence='c_v')
    coherence = cm.get_coherence()

    word_distributions = []
    for i in range(n_topic):
        word_distributions.append([list(word) for word in lda_model.show_topic(i, topn=20)])

    topic_distributions = []
    for i in range(len(data_tokens)):
        topic_distributions.append([list(topic_id) for topic_id in lda_model[corpus[i]][0]])

    doc_number = len(data_tokens)

    doc_dist = {}
    for i in range(n_topic):
        doc_dist.update({i: []})

    for i in range(doc_number):
        doc_dist[get_topic_dist_max(lda_model[corpus[i]][0])[0]].append(i)

    output = {"filecount": len(data_tokens),
              "coherence_value": float(coherence),
              "word_distributions": word_distributions,
              "topic_distributions": topic_distributions,
              "doc_dist": doc_dist,
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
