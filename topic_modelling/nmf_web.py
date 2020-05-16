import re

import gensim
import nltk
import numpy as np
from gensim.models import CoherenceModel
from nltk.tokenize import RegexpTokenizer
from sklearn.decomposition import NMF as sk_NMF
from sklearn.feature_extraction.text import TfidfVectorizer


def get_topic_dist_max(vector):
    dict_of_topics = dict(vector)
    for key in dict_of_topics.keys():
        dict_of_topics[key] = abs(dict_of_topics[key])
    maximum_topic = max(dict_of_topics, key=dict_of_topics.get)
    return maximum_topic, dict_of_topics.get(maximum_topic)


def NMF(corpus):
    my_stopwords = ['acaba', 'altmış', 'altı', 'ama', 'ancak', 'arada', 'aslında', 'ayrıca', 'bana', 'bazı', 'belki',
                    'ben',
                    'benden', 'beni', 'benim', 'beri', 'beş', 'bile', 'bin', 'bir', 'birçok', 'biri', 'birkaç',
                    'birkez',
                    'birşey', 'birşeyi', 'biz', 'bize', 'bizden', 'bizi', 'bizim', 'böyle', 'böylece', 'bu', 'buna',
                    'bunda', 'bundan', 'bunlar', 'bunları', 'bunların', 'bunu', 'bunun', 'burada', 'çok', 'çünkü', 'da',
                    'daha', 'dahi', 'de', 'defa', 'değil', 'diğer', 'diye', 'doksan', 'dokuz', 'dolayı', 'dolayısıyla',
                    'dört', 'edecek', 'eden', 'ederek', 'edilecek', 'ediliyor', 'edilmesi', 'ediyor', 'eğer', 'elli',
                    'en',
                    'etmesi', 'etti', 'ettiği', 'ettiğini', 'gibi', 'göre', 'halen', 'hangi', 'hatta', 'hem', 'henüz',
                    'hep', 'hepsi', 'her', 'herhangi', 'herkesin', 'hiç', 'hiçbir', 'için', 'iki', 'ile', 'ilgili',
                    'ise',
                    'işte', 'itibaren', 'itibariyle', 'kadar', 'karşın', 'katrilyon', 'kendi', 'kendilerine', 'kendini',
                    'kendisi', 'kendisine', 'kendisini', 'kez', 'ki', 'kim', 'kimden', 'kime', 'kimi', 'kimse', 'kırk',
                    'milyar', 'milyon', 'mu', 'mü', 'mı', 'nasıl', 'ne', 'neden', 'nedenle', 'nerde', 'nerede',
                    'nereye',
                    'niye', 'niçin', 'o', 'olan', 'olarak', 'oldu', 'olduğu', 'olduğunu', 'olduklarını', 'olmadı',
                    'olmadığı', 'olmak', 'olması', 'olmayan', 'olmaz', 'olsa', 'olsun', 'olup', 'olur', 'olursa',
                    'oluyor',
                    'on', 'ona', 'ondan', 'onlar', 'onlardan', 'onları', 'onların', 'onu', 'onun', 'otuz', 'oysa',
                    'öyle',
                    'pek', 'rağmen', 'sadece', 'sanki', 'sekiz', 'seksen', 'sen', 'senden', 'seni', 'senin', 'siz',
                    'sizden', 'sizi', 'sizin', 'şey', 'şeyden', 'şeyi', 'şeyler', 'şöyle', 'şu', 'şuna', 'şunda',
                    'şundan',
                    'şunları', 'şunu', 'tarafından', 'trilyon', 'tüm', 'üç', 'üzere', 'var', 'vardı', 've', 'veya',
                    'ya',
                    'yani', 'yapacak', 'yapılan', 'yapılması', 'yapıyor', 'yapmak', 'yaptı', 'yaptığı', 'yaptığını',
                    'yaptıkları', 'yedi', 'yerine', 'yetmiş', 'yine', 'yirmi', 'yoksa', 'yüz', 'zaten']
    replace_with_space = re.compile('[/(){}\[\]\|@,;]')
    remove_symbols1 = re.compile("[^0-9a-z_ğüşıöç .']")
    stopwords = nltk.corpus.stopwords.words('turkish')
    stopwords.extend(my_stopwords)
    remove_3chars = re.compile(r'\b\w{1,3}\b')

    def clean_text(text):
        """
            text: a string

            return: modified initial string
        """
        text = text.lower()
        text = replace_with_space.sub(' ', text)
        text = remove_symbols1.sub('', text)
        text = remove_3chars.sub('', text)
        text = ' '.join([word for word in text.split() if word not in stopwords])
        return text

    cleaned_data = [clean_text(news) for news in corpus]

    data_tokens = []
    tokenizer = RegexpTokenizer(r'\w+')
    for i in range(len(cleaned_data)):
        tokens = tokenizer.tokenize(cleaned_data[i])
        data_tokens.append(tokens)

    id2word = gensim.corpora.Dictionary(data_tokens)

    no_topics = 10

    vectorizer = TfidfVectorizer()
    A = vectorizer.fit_transform(cleaned_data)
    nmf_model = sk_NMF(n_components=no_topics, init='nndsvd')
    W = nmf_model.fit_transform(A)  # document topic distribution
    H = nmf_model.components_  # topic word distribution

    feature_names = vectorizer.get_feature_names()  # terms

    word_distributions = []
    word_count = 10

    for topic in range(len(H)):
        top_indices = np.argsort(H[topic, :])[::-1]

        doc_list = []
        for term_index in top_indices[0:word_count]:
            doc_list.append((feature_names[term_index], H[topic, term_index]))
        word_distributions.append(doc_list)

    topic_distributions = []
    for document in range(len(W)):
        topic_distributions.append([(topic, W[document][topic]) for topic in range(len(W[document]))])

    doc_number = len(W)
    topic_number = len(H)

    doc_dist = {}
    for i in range(topic_number):
        doc_dist.update({i: []})

    for i in range(doc_number):
        doc_dist[get_topic_dist_max(topic_distributions[i])[0]].append(i)

    nmf_topics = [[word[0] for word in topic] for topic in word_distributions]
    coherence = CoherenceModel(topics=nmf_topics, texts=data_tokens, dictionary=id2word).get_coherence()

    output = {"filecount": doc_number,
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
