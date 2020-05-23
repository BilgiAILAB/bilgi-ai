from topic_modelling.algorithms import topic_distance


def word_distribution(model, n_topic):
    word_distributions = []
    for i in range(n_topic):
        word_distributions.append([list(word) for word in model.show_topic(i, topn=20)])
    return word_distributions


def hdp_word_distribution(model, topics_n):
    word_distributions = []
    for topic_id in topics_n:
        word_distributions.append([list(word) for word in model.show_topic(topic_id, topn=20)])
    return word_distributions


def lda_topic_distribution(doc_number, model, corpus):
    topic_distributions = []
    for i in range(doc_number):
        topic_distributions.append([list(topic_id) for topic_id in model[corpus[i]][0]])
    return topic_distributions


def lda_doc_distribution(n_topic, doc_number, model, corpus):
    doc_dist = {}
    for i in range(n_topic):
        doc_dist.update({i: []})

    for i in range(doc_number):
        doc_dist[topic_distance.get_topic_dist_max(model[corpus[i]][0])[0]].append(i)
    return doc_dist


def lsi_topic_distribution(doc_number, model, corpus):
    topic_distributions = []
    for i in range(doc_number):
        topic_distributions.append([list(topic_id) for topic_id in model[corpus[i]]])
    return topic_distributions


def lsi_doc_distribution(n_topic, doc_number, model, corpus):
    doc_dist = {}
    for i in range(n_topic):
        doc_dist.update({i: []})

    for i in range(doc_number):
        doc_dist[topic_distance.get_topic_dist_max(model[corpus[i]])[0]].append(i)
    return doc_dist


def hdp_doc_distribution(n_topic, topics_n, doc_number, model, corpus):
    doc_dist = {}
    for i in range(n_topic):
        doc_dist.update({i: []})

    for i in range(doc_number):
        doc_dist[topic_distance.get_topic_dist_max(model[corpus[i]])[0]].append(i)

    last_dist = {}
    for element in topics_n:
        last_dist.update({topics_n.index(element): doc_dist.get(element)})
    return last_dist