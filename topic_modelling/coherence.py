from gensim.models import CoherenceModel


def coherence_value(model, tokens, dictionary):
    return CoherenceModel(model=model, texts=tokens, dictionary=dictionary, coherence='c_v').get_coherence()