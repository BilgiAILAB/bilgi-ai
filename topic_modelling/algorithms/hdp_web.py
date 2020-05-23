from gensim.models import HdpModel
from topic_modelling.algorithms import distributions, preprocess, coherence


def HDP(corpus):
    n_topic = 150
    cleaned_data, data_tokens, id2word, corpus = preprocess.preprocess(corpus=corpus)
    doc_number = len(data_tokens)
    hdp_model = HdpModel(corpus=corpus, id2word=id2word)
    coherence_v = coherence.coherence_value(model=hdp_model, tokens=data_tokens, dictionary=id2word)

    topic_distributions = distributions.lsi_topic_distribution(doc_number=doc_number, model=hdp_model, corpus=corpus)
    topics_n = []
    for document in topic_distributions:
        for topic_dist in document:
            if topic_dist[0] not in topics_n:
                topics_n.append(topic_dist[0])
            first_index = topic_distributions.index(document)
            second_index = topic_distributions[first_index].index(topic_dist)
            topic_distributions[first_index][second_index][0] = topics_n.index(topic_dist[0])
    word_distributions = distributions.hdp_word_distribution(model=hdp_model, topics_n=topics_n)
    doc_dist = distributions.hdp_doc_distribution(n_topic=n_topic, topics_n=topics_n, doc_number=doc_number, model=hdp_model, corpus=corpus)

    output = {"filecount": doc_number,
              "coherence_value": float(coherence_v),
              "word_distributions": word_distributions,
              "topic_distributions": topic_distributions,
              "doc_dist": doc_dist,
              "topics_n":topics_n,
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
