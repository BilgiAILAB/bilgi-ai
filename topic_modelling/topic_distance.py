def get_topic_dist_max(vector):
    dict_of_topics = dict(vector)
    for key in dict_of_topics.keys():
        dict_of_topics[key] = abs(dict_of_topics[key])
    maximum_topic = max(dict_of_topics, key=dict_of_topics.get)
    return maximum_topic, dict_of_topics.get(maximum_topic)
