import nltk
import re
from nltk.tokenize import RegexpTokenizer
import gensim
from topic_modelling.algorithms.pos_tagger import tag
import os


def preprocess(corpus):
    file = open(f"{os.path.dirname(__file__)}/turkce-stop-words.txt")
    stops = file.readlines()
    replace_with_space = re.compile('[/(){}\[\]\|@,;]')
    remove_symbols1 = re.compile("[^0-9a-z_ğüşıöç .']")
    stopwords = nltk.corpus.stopwords.words('turkish')
    stopwords.extend(stops)
    remove_3chars = re.compile(r'\b\w{1,3}\b')

    def clean_text(text):
        """
            text: a string

            return: modified initial string
        """
        valid_characters = 'abcçdefgğhıijklmnoöpqrsştuüvwxyzQWERTYUIOPĞÜASDFGHJKLŞİZXCVBNMÖÇ1234567890 '
        text = ''.join([x for x in text if x in valid_characters])
        text = " ".join([word_tag[0] for word_tag in list(tag(text)) if word_tag[1]=='Noun_Nom' or word_tag[1]=='Adj'])
        lower_map = {
            ord(u'I'): u'ı',
            ord(u'İ'): u'i',
            ord(u'Ö'): u'ö',
            ord(u'Ü'): u'ü',
            ord(u'Ş'): u'ş',
            ord(u'Ğ'): u'ğ',
        }
        text = text.translate(lower_map)
        text = text.lower()
        #text = replace_with_space.sub(' ', text)
        #text = remove_symbols1.sub('', text)
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

    return cleaned_data, data_tokens, id2word, corpus