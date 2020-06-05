import plotly.graph_objects as go
from gensim.models import LdaModel

from topic_modelling.algorithms import distributions, preprocess, coherence


def lda_optimum_coherence(corpus, start, end, step):
    cleaned_data, data_tokens, id2word, corpus = preprocess.preprocess(corpus=corpus)
    topic_numbers = []
    coherence_values = []

    for num_topics in range(start, end + 1, step):
        lda = LdaModel(corpus=corpus,
                       id2word=id2word,
                       num_topics=num_topics,
                       random_state=100,
                       update_every=1,
                       passes=10,
                       alpha='auto',
                       per_word_topics=True,
                       minimum_probability=1e-8)
        coh = coherence.coherence_value(model=lda, tokens=data_tokens, dictionary=id2word)
        topic_numbers.append(num_topics)
        coherence_values.append(coh)
    fig = go.Figure(data=go.Scatter(x=topic_numbers, y=coherence_values))
    return fig


def LDA(corpus, n_topic):
    cleaned_data, data_tokens, id2word, corpus = preprocess.preprocess(corpus=corpus)
    doc_number = len(data_tokens)
    lda_model = LdaModel(corpus=corpus,
                         id2word=id2word,
                         num_topics=n_topic,
                         random_state=100,
                         update_every=1,
                         passes=10,
                         alpha='auto',
                         per_word_topics=True,
                         minimum_probability=1e-8)
    coherence_v = coherence.coherence_value(model=lda_model, tokens=data_tokens, dictionary=id2word)
    word_distributions = distributions.word_distribution(model=lda_model, n_topic=n_topic)
    topic_distributions = distributions.lda_topic_distribution(doc_number=doc_number, model=lda_model, corpus=corpus)
    doc_dist = distributions.lda_doc_distribution(n_topic=n_topic, doc_number=doc_number, model=lda_model,
                                                  corpus=corpus)

    output = {"filecount": doc_number,
              "coherence_value": float(coherence_v),
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
