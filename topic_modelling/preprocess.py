import nltk
import re
from nltk.tokenize import RegexpTokenizer
import gensim

def preprocess(corpus):
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

    return cleaned_data, data_tokens, id2word, corpus