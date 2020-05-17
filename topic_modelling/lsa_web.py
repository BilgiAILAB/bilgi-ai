import re

import gensim
import nltk
from gensim.models import CoherenceModel, LsiModel
from nltk.tokenize import RegexpTokenizer


def get_topic_dist_max(vector):
    dict_of_topics = dict(vector)
    for key in dict_of_topics.keys():
        dict_of_topics[key] = abs(dict_of_topics[key])
    maximum_topic = max(dict_of_topics, key=dict_of_topics.get)
    return maximum_topic, dict_of_topics.get(maximum_topic)


def LSA(corpus, n_topic):
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

    corpus = [id2word.doc2bow(doc) for doc in data_tokens]

    lsi_model = LsiModel(corpus=corpus, num_topics=n_topic, id2word=id2word)

    cm = CoherenceModel(model=lsi_model, texts=data_tokens, dictionary=id2word, coherence='c_v')
    coherence = cm.get_coherence()

    word_distributions = []
    for i in range(n_topic):
        word_distributions.append([list(word) for word in lsi_model.show_topic(i, topn=20)])

    topic_distributions = []
    for i in range(len(data_tokens)):
        topic_distributions.append([list(topic_id) for topic_id in lsi_model[corpus[i]]])

    doc_number = len(data_tokens)

    doc_dist = {}
    for i in range(n_topic):
        doc_dist.update({i: []})

    for i in range(doc_number):
        doc_dist[get_topic_dist_max(lsi_model[corpus[i]])[0]].append(i)

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
