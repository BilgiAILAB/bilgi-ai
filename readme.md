# How to install

1. [Download](https://github.com/ibrahim-dogan/cmpe-graduation-project/archive/master.zip) or clone repository with command-line:
```
git clone https://github.com/ibrahim-dogan/cmpe-graduation-project.git
```
2. Install requirements
```
cd cmpe-graduation-project
pip install -r requirements.txt
```
3. Run this commands to initialize DB file
```
python manage.py migrate
python manage.py collectstatic
```
4. You can run the server now! 
```
python manage.py runserver
```
5. Go to [http://127.0.0.1:8000/](http://127.0.0.1:8000/) to see the website.

# Read the Report

You can read the report from [here](https://docs.google.com/document/d/1afgEeB5FLSDd_L277uXEU_pVHAQZpND2IAH9s4Sl4pc/edit?usp=sharing) 

# Preview (just for quick view to the report)


WEB APPLICATION BASED ON TOPIC MODELING AND DOCUMENT SIMILARITY

by

İbrahim Doğan (11512112)

Hanifi Enes Gül (115200085)

Simge Erek (115200023)

CMPE 492 Senior Design Project

2020

  
  

Course Teacher: Pınar Hacıbeyoğlu

Advisor: Tuğba Yıldız

  

Computer Engineering Department

Faculty of Engineering and Natural Science

Istanbul Bilgi University![](https://lh3.googleusercontent.com/7rf4v_2WJZxltEXKKyaa11hEEuspGeUkhCHkNjY2OABdbQYifq3n802PNGjxFIkKBbDoW0IhdbzKzCDSwRX8GlU9Y9nwUln_bwp_OZK4V1cQ1rkdSZ7Y4a-WP10HiSZxERutrvLN)

Table of Contents
Abstract	5
1.Introduction	6
2. Literature Search	7
3.Methodology	10
3.1 Datasets	10
3.2 Methods	10
4. Experimental Setup	18
4.1 Document Similarity	18
4.2 Topic Modeling	22
5. Experimental Results	28
      5.1 Topic Modeling	28
      5.2 Document Similarity	29
6. Web Platform	32
6.1 Workflow	33
7. Future Works	34
REFERENCES	35
APPENDIX	37

  
Table of Figures

  

Figure 1: Skip gram and CBOW structure [17]

Figure 2: Database Schema in UML format

 

Table of Tables

Table 1: Comparison of related websites

Table 2 : TF-IDF Model

Table 3 : Topic Modeling Method Characteristics

Table 4: Experimental Result of Topic Modeling Methods where n is news title (9)

Table 5 : 5 most similar news are shown for TF-IDF approaches

Table 6 : 5 most similar news are shown for Word2Vec approaches

  

# Abstract

With the development of technology, speed of accessing data has increased. This speed allows human to produce more data. Because there are too many to read, human understood that this process should be automated and started to research. Researchers found that the big collection of data can processed with mathematics and statistics, there are many algorithms designed to measure similarity between texts and classify texts. This article focuses on algorithms and conducted experimental studies on various data sets. The aim of this project is to provide the user with wide and different options, so that it can easily perform discovery and analysis with the similarity between documents or topic modeling through the web interface. The flow as follows, user is uploading its data collection, select which type of work to done either text similarity or topic modeling, then the model is generating the outcome.


# 1.Introduction

In the past, research on a subject was possible with books and printed sources. As the internet became widespread, our access to information became easier and this led to an increase in the number of resources. Such many resources made it impossible for human beings to read them all.

Before writing an article on a topic, researchers read several articles on the topic they are researching. However, each article may not contain the topic that the researcher would like to address directly, requiring the researcher to read more articles, i.e. effort and time, to find the specific topic he / she is looking for.

Studies in the field of artificial intelligence and natural language processing showed that mathematics and statistics produced successful results in examining such collections of text.

The main purpose of this study is to provide convenience to the researchers with the tool we have prepared. Topic Modeling methods will be used for the clustering process and Document Similarity methods will be used to filter similar articles-texts.

There are many natural language processing applications that calculating similarity between text has major significance such as: book recommendation [1], news categorization [2], essay scoring [3]. Document similarity is especially useful to get an overview of a corpus that is new to user. Product of this project includes two section which are -highly related with each other- topic modeling and document similarity. Topic models can connect words with similar meanings and distinguish between uses of words with multiple meanings. Topic models are especially useful when dealing with large corpus, however it can be used on a smaller corpus as long as there is homogeneity in word usage. Moreover, it can be used out certain documents or highlight others and can be used to show trends in topics over time.

Central task of this project is to giving opportunity to user to try different topic models-approaches and methods that can measure similarity between documents. Moreover, a web application is built based on these topic models and methods to serve anyone who needs to explore texts efficiently in a short time. Python3 was used because of its wide library support and the wide range of algorithms we use. The Django web framework was used because the python codes written with the support of this library can be easily integrated.

# 2. Literature Search

There are many different approaches to measuring, comparing and evaluating the similarity between documents, such as corpus-based, knowledge-based, and string-based similarity. Many articles have been published examining these approaches. Two different studies by Mamdouh Farouk [4] and Weal H. Gomao & Aly A. Fahmy [5] have examined three different approaches for sentence similarity and have shown that combined approaches work better. At the same time, Aditi G. et.al also discussed the features, performances, advantages and disadvantages of the two similarity approaches to help choose the best similarity approach [6]. Didik Dwi Prasetya et.al compared the four different text similarity approaches and showed that the semantic similarity was more rational [7].

Similarity measures and text representation methods have a major impact on the performance of text similarity. P. Sitikhu et.al compared three different approaches to measure the semantic similarity between short texts [8]. They showed that the TF-idf-cosine similarity approach had higher accuracy value. Erwan Morean et.al showed that combining various similarity measures would provide better performance [9]. Radha Mothukuri et.al also proposed a new measure of similarity to reduce feature set noise [10].

The first step researched in topic modeling is preprocessing. The most basic steps of cleaning in similar studies are: parsing stopwords and lowercasing the text. According to studies, with using part-of-speech tagging; while the success of the work is not negatively affected, it allows the models to work faster because it allows us to process with fewer words. Martin&Johnson proved that nouns only approach not only works better than without pos and lemmatised documents but also works in significantly short time [11]. Beside part-of-speech, while the basic pre-process is making a minor difference for the LDA model, it is of great importance in the word embeddings approach. It is shown that it affects coherence value around 0.10 when using word2vec approach [12].

  

There are many studies where several of the conventional topic modeling algorithms are compared in detail. Thanks to Kherwa & Bansal, it is possible to explore the working principles and comparisons of LDA, LSA and NMF methods [13]. Kim S. et al. combined word2vec and Latent Semantic Analysis model for topic modeling to analyze blockchain technology trends [14].

  

In the light of foregoing, it is possible to quantify degree of relatedness between two pieces of texts in a wide variety ways such as combining different embeddings (BoW, tf-idf, CBoW, pre-trained word embeddings: word2vec, GloVe, fasttext etc.), clustering algorithms, topic modeling algorithms (lsa, lda, nmf, plsa, hdp, dtm etc.), similarity methods (jaccard, cosine and euclidean similarities). However, we cannot see this diversity online. We see that the varieties are more likely to detect plagiarism ın the syntactic similarity side. Although there are websites and API’s such as cortical.io, dandelion.eu, twinword.com; their purpose is not giving opportunity to user to explore given documents. Also they includes many restrictions as it can be seen in Table 1 below.

  

  

Semantic Meaning Comparison

  

Character Limit

  

For End User

  

Multiple File Input

cortical.io

Yes

No

Yes

No

dandelion.eu

Yes

Yes

No (API)

No

twinword.com

Yes

No

No (API)

No

Table 1: Comparison of related websites

  
  

For these reasons, the aim of this project is to provide the user with wide and different options, so that it can easily perform discovery and analysis with the similarity between documents or topic modeling through the web interface.

  
  

# 3.Methodology

## 3.1 Datasets

The dataset used for experimental study is taken from “Kemik” research group of Yıldız Technical University [15]. The dataset contains Turkish news collected from 41,992 newspapers in total, collected from 13 separate news headlines. This headlines consists of world, economy, general, daily, culture and art, magazine, planet, health, politics, sports, technology, Turkey and life. More than 1000 individually recorded articles in each subtitle are in txt format. Also another dataset is created for this study. This dataset includes 27 books in 5 different areas which are psychology, philosophy, history, science and art. All books are converted from epub to txt file extension.

  

## 3.2 Methods

  

3.2.1 Document Similarity

Document similarity is the natural language processing study in which two documents are compared to find the similarity. In this natural language processing process, examining the similarity between words is the most essential part to find the similarity of sentences, texts, documents. There are two main types of similarity that can be used for the evaluation of similarity between words: lexical and semantic similarities. If words have a similar character sequence, words are lexically similar, if words can be used close to each other, semantically similar.

  

The studies classifying text similarity methods [5,16] collect the methods in two classes: lexically and semantically. String-based algorithms are used to measure lexical similarity and corpus-based and knowledge-based algorithms are used to measure semantic similarity.

  

One aim of this study is to provide many different methods to find document similarities. To find the similarity, the feature vectors of the documents are determined and the distance between the features is calculated. Cosine similarity, Euclidean distance, Manhattan distance are used to calculate document similarity.And Tf-Idf, Word2Vec are presented as different methods to create features from documents.

Methods to calculate the semantic similarity between documents; Cosine similarity using Tf-Idf, Euclidean distance using TF-Idf, Manhattan Distance using Tf-Idf, Cosine Similarity using Word2Vec, Euclidean Distance using Word2Vec, Manhattan Distance using Word2Vec.

  

3.2.1.1 TF-IDF

TF-IDF is a numerical statistics that aims to show the importance of a word in a document. It consists of two numerical statistics called Term Frequency and Inverse document frequency.

Term frequency is the number of times a term appears within a document. In longer documents, the number of times a term is higher than in short documents, therefore the frequency of term t in document d is divided by the total number of terms in the document d for normalization.

  

tf(t,d) = number of times term t appears in document dtotal number of terms in the document d

  

Inverse document frequency is a weight that shows how often a term t is used in document collections D. The less use of the term in documents, the higher the idf score. A high score increases the importance of the term.

  

idf(t,D) = logtotal number of document collection Dnumber of documents in which term t appears

  

The most common terms are not the most informative terms. Rarely occurring terms in documents are more meaningful and informative for the document collection. Therefore, the term frequency tf(t, d) is converted to the tf-idf weight  with idf(t, D).

  

tfidf(t,d,D) = tf(t,d)idf(t,D)

As an example, consider these three document below:

-   d1: sevimli beyaz kedi
    
-   d2: yaramaz çocuk
    
-   d3: yaramaz kedi
    

The TF-IDF model is applied to the given examples. The words in the documents are listed and the tf value is calculated for the words in each document. Then the idf value is found for each word. The TF-IDf model is shown in the table 2.

WORD

TF (d1)

TF (d2)

TF (d3)

IDF

TF-IDF (d1)

TF-IDF (d2)

TF-IDF (d3)

sevimli

1/3 = 0.333

0/2 = 0

0/2 = 0

log(3/1) =0.398

0.133

0

0

beyaz

1/3 = 0.333

0/2 = 0

0/2 = 0

log(3/1) =0.398

0.133

0

0

kedi

1/3 = 0.333

0/2 = 0

1/2 = 0.5

log(3/2) =0.135

0.045

0

0.068

yaramaz

0/3 = 0

1/2 = 0.5

1/2 = 0.5

log(3/2) =0.135

0

0.068

0.068

çocuk

0/3 = 0

1/2 = 0.5

0/2 = 0

log(3/1) =0.398

0

0.199

0

Table 2 : TF-IDF Model

  

3.2.1.2 Word2Vec

Word2vec is an prediction-based word embedding model used to express words as vectors and to calculate the distance between them. It is one of the most generally utilized types of word embeddings. The word2vec model scans a document in a structure called a window and outputs a vector formed according to the target word in the window.

  

There are two models called CBOW and skip-gram that word2vec uses. The structure of the models is shown in the Figure. The main difference in these two models is based on the methods of getting input and output.[17].

  

![](https://lh4.googleusercontent.com/BMBLIHnW1sewBnmUdg9HvyS8PcRa2jyEM0vAa_jkqr0O5EpY2mpga0d5Prbv2SNUDSYTYMlPb47TSxhJdN57BxrbPKIG6F5L8mqrUOYh3kiptxbRM-CCv5nL4OvYndasTdK3Or5T)

Figure 1: skip gram and CBOW structure [17]

  

In the Continuous Bag of Words model, words that are not in the center of window size are taken as input and the word in the center is estimated as output. Unlike CBOW, in the Skip-Gram model, the word at the center of the window size is taken as input, and words that are not at the center of the window size are estimated as output.

  

Gensim, an open source library for unsupervised topic modeling and natural language processing, provides word2vec models trained by different data sets such as wikipedia, GloVe, google news, ConceptNet, twitter.

  

3.2.1.3 Similarity Measures

Cosine similarity is the most popular similarity measure used for text similarity. It measures the cosine of the angle between two vectors and its value is between 0 and 1. Cosine similarity of similar vectors is close to one.A and B are two vectors, the cosine similarity is calculated by dot product and magnitude as follows:

  

cos() = A.BAB

  

Euclidean distance is the ordinary distance between two points in the plane of 2 or 3. If the distance is 0, the two points are the same. X=[x1, x2, . . . ,xn] and Y=[y1, y2, . . . ,yn] are two vectors, euclidean distance is defined as follows:

  

X-Y=i=1n (xi-yi)2

  

Manhattan distance is equal to the length of the shortest line connecting the two points.

X=[x1, x2, . . . ,xn] and Y=[y1, y2, . . . ,yn] are two vectors, Manhattan distance is defined as follows:

X-Y=i=1nxi-yi

  

3.2.2 Topic Modeling

Topic modeling is an unsupervised machine learning method which is used for clustering documents via retaining word patterns. It is useful technique to analyze large amounts of data. With the help of topic modeling it is effortless to understand-explore topical patterns among documents. It is basically a tool that can infer group of words -which is called topic- from a corpus. Furthermore, using this widely used NLP technique makes it possible to organize, summarize texts. There are many methods to do topic modeling some of which are described below.

  

3.2.2.1 Topic Modeling Methods

Topic modeling is an approach that allows us to cluster text documents and see the hidden titles of each document called the topic. There are many methods developed for this approach, but each has its own success and failure. There are many different methods and approaches that are statistical, semantic, effective in short documents and effective in long documents. There are even methods that we can identify topic trends with time data. However, since we do not receive time data in this study and we expect a data set consisting of documents with a minimum length of 200-300 words from the user; We have analyzed methods that meet these criteria and integrated them into our website.

  

3.2.2.1.1 LATENT DIRICHLET ALLOCATION (LDA)

LDA is based on three-level hierarchical Bayesian model [18]. LDA is a generative model that tries to mimic what the writing process is. So, it tries to generate a document on the given topic. It can also be applied to other types of data. There are tens of LDA based models including: temporal text mining, author- topic analysis, supervised topic models, latent Dirichlet co-clustering and LDA based bioinformatics [19], [20].

The order of the words in the document is not significant for this method. In other words, the bag of words (BOW) approach has been completely adopted. Posterior distribution of latent variables behind LDA is shown below:

![](https://lh5.googleusercontent.com/l5Xp7ASoIIudtu39S1RGlUnPJW8f5KHyDshphVvybgN-nCx-7xQ-kB2RWoNGrqUr28mWSeQgOaojSTIIMnc5nYaSvpNfQK_-5-jalbn5CuU0i9Boa3UW4ztTs8b5X3VQOlp5N5cl)

Calculations done for each document:

(a) Distribution on topics θ~Dir(α). Dir stands for dirichlet distribution and α is a scaling parameter.

(b) For every word included in document

1.Draw a specific topic z~ multi(θ)

2. Draw a word w~βz. [13]

  

By using LDA not only exploring-extracting topics from the given documents is possible but also topic distributions over per document and word weights over topics can be calculated.

As it is mentioned before methods includes weaknesses. Although LDA is very popular, it doesn’t mean that it fits perfect for every dataset. On the contrary, we observe that LDA gives poor results when used for short documents. Since it is a probabilistic approach, it depends on word co-occurrences. Therefore, success of this model is highly related with given dataset.

  

3.2.2.1.2 LATENT SEMANTIC ANALYSIS (LSA)

Latent semantic analysis (a.k.a. Latent semantic index-LSI) is the most popular Corpus based similarity technique. LSA is based on the assumption that closely related words coexist in similar texts[16]. Created semantic relations are based upon the given corpus. Organizing and clustering texts semantically is the purpose of LSA; to achieve this purpose and find related words LSA represents texts as a vector [13].

  

Working principle of LSA is firstly creating a term-document matrix. The key move of LSA is applying Singular Value Decomposition (SVD) to term-document matrix. If we call term-document matrix as A; then SVD creates three sub-matrix.

  

![](https://lh5.googleusercontent.com/F8MJBzSxHKu2ZJACdhmPDCzQm3uGZW9xAwPomYlaYozhHS8SfGbtXevPsPM-NH7DOGPGeEcxOgBrNwIsbRyeltI2gwVoFUeBvia9U9fumqCDxN0Qo8U6xtzCZ_HUJr1iTiznubIk)

  

These three sub-matrix are U, ∑ and VT. U represents orthogonal matrix, ∑ stands for diagonal matrix and finally VTis the transpose of V which is an orthogonal matrix. With the help of these three matrix we extract the word matrix for topic, topic strength and document matrix for topic in given order.

  

3.2.2.1.3 Hierarchical Dirichlet Process (HDP)

Hdp is a Bayesian nonparametric model which is not popular as much as other methods mentioned. Nevertheless, it is known to be a very successful technique when using large datasets. What makes this method stand out is that, unlike other methods, how many topics are created doesn't need to be taken as parameters. Hdp is able to infer the number of topics from the given data. Given a document collection, posterior inference is used to calculate the number of topics and to characterize their distribution [21]. To infer topic number just by getting data, inference algorithms requires passing through all data multiple times. This means a time and processing cost that we accept in return for not requiring the number of topics. Since it is preferred especially in large datasets, it can be said that it takes a little longer to work than other algorithms.

  
  

3.2.2.1.4 NON-NEGATIVE MATRIX FACTORIZATION (NNMF-NMF)

Nmf is a dimension reduction technique which decomposes high-dimensional vectors into lower-dimensional representation. Negative components are contradict physical reality. Therefore, it is a representation of data confined to nonnegative vectors[13]. NMF creates two matrices with using a matrix which is created by applying td-idf, count vectorizing, binomial vectorizers etc. One of the created matrix represents the topics which are found and other one is word weights for those topics. In practice while working with short documents (eg tweet), NMF is generally expected to work more successfully than LDA and has been observed.

  

3.2.2.1.5 Word Embeddings

Beside statistical methods, with the usage of word embeddings we can deduct the semantic relations between words. All methods explained before for topic modeling don’t include any data other than given corpus. Whether they are based on words or topic, all relation between words, topics and documents are inferred just by using document. In these conventional approaches where the similarity is understood by the co-occurence of the words in the document, they are not able to relate the words we know to be close in our natural language (especially in small datasets). Despite their success in many issues, these models also fail on short documents because word repetitions are very limited in short documents. The document-topic distributions and the topics are less semantically coherent created because they cannot associate the true meaning of the words [22].

  

Text clustering and topic modeling have quite similar purpose to understand the text data. When they are used mutually, increasing result is a natural expectation. In this approach, to represent each document without losing semantic value of words, each document is represented with vectors using word2vec (doc2vec). Created vectors are clustered with k-means algorithm. Lastly, LDA is applied to each cluster. All in all, these steps makes it possible to get topics and distributions more coherently and semantically related.

  
  
  

Method Name

Characteristics

Latent Semantic Analysis (LSA)

-A theory/method used to extract the contextual usage meaning of words using statistical computation which is applied to a large corpus.

-It can get topic if there are any synonym words.

-Uses SVD to condense a Term-Document Matrix.

-It is hard to obtain the number of topics

Probabilistic Latent Semantic Analysis (pLSA)

Same as LSA but replaces SVD with a probabilistic method.

-Handles polysemy.

Latent Dirichlet Allocation (LDA)

-Identical to pLSA except that in LDA the topic distribution is assumed to have a sparse Dirichlet prior. The sparse Dirichlet priors encode the intuition that documents cover only a small set of topics and that set of topics use only a small set of words frequently.

Hierarchical Dirichlet Process (HDP)

-Infers number of topic itself from the corpus.

-To infer topic passes through corpus multiple times which brings computation and time cost.

Non-Negative Matrix Factorization (NMF)

-Uses matrix factorization

-Similar with LSA

-Assumes there are no negative numbers in original matrix.

Word Embedding Models (WEMs)

-WEMs do the opposite of other topic models.

-Try to ignore information about individual documents so that you can better understand the relationships between words.

Table 3 : Topic Modeling Method Characteristics

  
  
  
  

# 4. Experimental Setup

## 4.1 Document Similarity

A document is selected from the uploaded documents by the user and the similarity or distance of the selected document with other documents is calculated. The user is provided six different options; Euclidean distance using Tf-idf, Cosine similarity using Tf-idf ,Manhattan distance using Tf-idf, Euclidean distance using Word2Vec, Cosine similarity using Word2Vec, and Manhattan distance using Word2Vec.

  

4.1.1 Preprocessing

Documents need to be cleaned for better performance of methods before being represented by Tf-Idf and Word2vec feature vectors.In the document similarity section of this study, preprocessing consists of six components:

-   removing whitespace
    
-   removing numbers
    
-   removing punctuations
    
-   convert all characters into lowercase
    
-   removing stopwords
    
-   tokenizer
    

The last step is not in the similarity methods to be used TF-IDF. The TfidfVectorizer structure does this step on its own.

  

4.1.2 Document Similarity using TF-IDF

The documents are preprocessed and converted into tf-idf vectors. These vectors consist of matrix, the size of which is in the number of documents and the number of words in each document. It is a matrix containing tf-idf weight of every word of each document. This matrix is ​​used as feature. It provides three different methods for document similarity using cosine similarity, euclidean distance and manhattan distance separately: Cosine similarity using Tf-Idf, Euclidean Distance using Tf-Idf and Manhattan Distance using Tf-Idf.

  

The value of a word in the analyzed document is measured by TF-IDF analysis. TfidfVectorizer is loaded from the Sklearn library to convert documents into tf-idf matrix. Sklearn is the most widely used library in machine learning. The vectorizer initialise and then call fit and transform to calculate the TF-IDF score for the text that preprocessed and given by user.

  

from  sklearn.feature_extraction.text import  TfidfVectorizer

tfidf = TfidfVectorizer()

tfs = tfidf.fit_transform(cleantext)

  

Dataframe from pandas library is used to process the data more easily. Pandas is one of the most popular libraries that provide high-performance, easy-to-use data structures and data analysis tools. A dataframe is created using tf-idf scores.

import  pandas  as  pd

df = pd.DataFrame(tfs.toarray(),columns=tfidf.get_feature_names())

The dataframe consisting of tfidf scores is used as a feature.

Cosine Similarity using Tf-Idf: Cosine similarity is imported from Sklearn library. The cosine similarity is calculated using the dataframe consisting of tf-idf scores. A new dataframe is created with the calculated similarity values.

from  sklearn.metrics.pairwise import  cosine_similarity

cossim = cosine_similarity(df)

cossimdf = pd.DataFrame(cossim)

Euclidean Distance using Tf-Idf:. Euclidean distance is imported from the Sklearn library. Euclidean distance is calculated using the dataframe consisting of tf-idf scores. A new dataframe is created with the calculated distance values.

from  sklearn.metrics.pairwise import  euclidean_distances

eucsim = euclidean_distances(df)

eucsimdf = pd.DataFrame(eucsim)

Manhattan Distance using Tf-Idf: Manhattan distance is imported from the Sklearn library. Manhattan distance is calculated using the dataframe consisting of tf-idf scores. A new dataframe is created with the calculated distance values.

from  sklearn.metrics.pairwise import  manhattan_distances

mansim = manhattan_distances(df)

mansimdf = pd.DataFrame(mansim)

4.1.3 Document Similarity using Word2vec

Pre-trained Word2vec model[23] that consist 412.467 words and 300 dimension was loaded with gensim.Documents are pre-processed. The vector value of each word in the documents is calculated using the word2vec model. The words that do not exist in the vocabulary of the model are ignored. The average of the word vectors of each document is calculated and a new vector is created. This vector is used as a feature for the document.It provides three different methods for document similarity using cosine similarity, euclidean distance and manhattan distance separately: Cosine similarity using word2vec, Euclidean Distance using word2vec and Manhattan Distance using word2vec.

from  gensim.models import  Word2Vec, KeyedVectors

model = KeyedVectors.load_word2vec_format("trmodel", binary=True)

The vector value of each word in the documents is calculated. Words that are not in the vocabulary of the model are ignored and the mean of the word vectors of each document is calculated. All documents is converted into vector.

def  document_vector(doc):

doc = [word for word in  doc if  word in  model.vocab]

return  np.mean(model[doc], axis=0)

These vectors created with word2vec are used as feature.

Cosine Similarity using word2Vec: The cosine similarity of two document vectors created with the word2vec model is calculated.

def  cosine_similarity(vector1, vector2):

dot_product = sum(p*q for  p,q in  zip(vector1, vector2))

magnitude = math.sqrt(sum([val**2 for  val in  vector1])) * math.sqrt(sum([val**2 for  val in  vector2]))

if  not  magnitude:

return  0

return  dot_product/magnitude

Euclidean Distance using word2vec:. The euclidean distance of two document vectors created with the word2vec model is calculated.

def  Euclidean(vector1, vector2) :

return  math.sqrt(sum(math.pow((v1-v2),2) for v1,v2 in  zip(vector1, vector2)))

Manhattan Distance using word2vec: The manhattan distance of two document vectors created with the word2vec model is calculated.

def  manhattan_distance(vector1, vector2):

return  sum(abs(e1-e2) for  e1, e2 in  zip(vector1,vector2))

## 4.2 Topic Modeling

  

4.2.1 Preprocessing

In practice, conventional topic modeling algorithms give more successful results when using large numbers of long document sets. The reason for this is that statistical approaches draw from repetitions of using words. For example, short texts such as tweet will not be sufficient for healthy work of conventional topic modeling algorithms. Although this study is not targeted for short documents, for the purpose of this study, since it contains conventional topic modeling algorithms, the data received from the user should be cleaned up as meaningful as possible so that statistical approaches can work properly. In this study, using NLTK and regex: words are converted to lowercase, sentences are free of symbols and only alphanumeric characters are kept. Because of the special case of Turkish, string translate method was used to convert some letters (eg I, İ, I etc.). conjunctions and two-letter words have been deleted. Besides, it has been observed in the researches that while topic modeling approaches using only nouns do not lose their success, at the same time, algorithms give much faster results since they are processed with fewer words[24]. Since short texts were not expected from this intended user, this approach did not cause bad situation for statistical models, on the contrary, it was found that much more smooth topics were obtained because the verbs such as frequently said "said", "expressed" would not participate in the use. In this study, since we will not want the user to wait too long on our website, this correct working and fast method was adopted. Only nouns and adjectives were processed using Turkish-pos-tagger, which Onur Yılmaz shared as an open source and successfully created part of speech tags with 94 percent success. In addition to these, Turkish stopwords have also been deleted. Cleared texts are tokenized with regexptokenizer because all approaches in the system use bag of words. Stemming or lemmatization is not used since there is no successful python libraries for Turkish language. Those libraries were not used because they do not work successfully enough to compensate for the lost time in return for their use.

  

4.2.2 Topic Modeling Methods

4.2.2.1 LDA

LdaModel was used from the Gensim library created by Radim Rehürek due to its implementation and simplicity of use. For the operation of the Lda model, tf-idf etc. approaches are not required, but since the BoW approach is adopted, the document-term matrix is sufficient for this model. It is possible to create document-term matrix and dictionary of tokens from given data tokens is done just by doing:

  

id2word = gensim.corpora.Dictionary(data_tokens)

  

corpus = [id2word.doc2bow(doc) for doc in data_tokens]

  

To create Lda model only topic number is needed except id2word and corpus which is taken from the user and explained in next sections. Although, there are many parameters that can be changed, since target user is not expected to know programming or have knowledge about topic modeling models; the opportunity to choose or play with parameters is not offered to the user on the website. It was preferred to leave it at its default settings, since the parameters are not an optimum accuracy value and can vary depending on the data they are applied to. In Lda model, only minimum_probability parameter is set to 1e-8. This is because, in document-topic distribution section user is wanted to see all probabilities even if there is a small percentages. These topic-word and document-topic distributions are done by:

  

word_distributions = distributions.word_distribution(model=lda_model, n_topic=n_topic)

  

topic_distributions = distributions.lda_topic_distribution(doc_number=doc_number,

model=lda_model,

corpus=corpus)

  

Where n_topic is the number of topic which is taken from the user. These distributions are stored to be shown user in explore stages.

  

4.2.2.2 LSA

LSA a.k.a. LSI is used by getting LsiModel from gensim library. Thanks to gensim it is also as easy as LDA to implement and use. After getting the data from the user -which is document collection- and the preprocess steps, word dictionary and corpus is created same way it is done in LDA. Compulsory parameters to create LSI is same as LDA which are corpus, id2word and num_topics which is number of topics to create. It is done simply by:

  

LsiModel(corpus=corpus, num_topics=n_topic, id2word=id2word)

  

As it is mentioned before, other optional parameters has no optimum value regardless from the dataset. Since it is highly dataset dependent, it is preferred to set as default. Getting topic-word and document-topic distributions are not differ from the way it is done in LDA.

  
  
  
  

4.2.2.3 HDP

HdpModel is also imported from gensim library. The major and only difference in creating HDP model is that HDP doesn’t require topic number. It just takes two compulsory parameters which are corpus and id2word. As it is explained under section 3.2.1 HDP infers topic numbers automatically from the given corpus. As default HDP has 150 potential topics. It fills and creates topics up to 150. In small data sets it is possible to observe that after small amount of topics, word weights in topic approaches to 0 which means these topics are not mainly find places in documents. Therefore the way getting distributions is slightly different. For example HDP may create 30 topics but 10 of them might be useless and have not weight on other document. In this situation user may not want to see this kind of ineffective topics. To avoid confusion and keep it simple, our approach was to only showing word distributions of topics which have weights on documents and have found place in document-topic distribution.

  

4.2.2.4 NMF

NMF is imported from sklearn (scikit-learn) library. It is possible to use NMF with different matrix representations using tf-idf vectorizer, countvectorizer, binomial vectorizer etc. In this study tf-idf vectorizer is used to create matrix. As it is explained before, NMF creates two different matrix by using that matrix created by tf-idf. One represents, document-topic distribution and another one represents topic-word distribution. NMF needs term-document matrix and n_components which is number of topics desired.

  

A = vectorizer.fit_transform(cleaned_data)

nmf_model = sk_NMF(n_components=n_topic, init='nndsvd')

W = nmf_model.fit_transform(A) # document topic distribution

H = nmf_model.components_ # topic word distribution

  

init parameter is set as as ‘nndsvd’. Nonnegative Double Singular Value Decomposition (NNDSVD) works only if desired topic number is smaller or equal to given document number. In other models it is possible to get topic more than the given document number. Also, nndsvd works slightly slower than it’s alternatives such as ‘nndsvda’ and ‘nndsvdar’. Despite these, nndsvd gives more accurate results and better for sparseness. In distributions negative values can be seen. Also unlike other methods, document-topic distribution is not represented as percentages. Evaluation of distribution is based on getting the absolute of the given distribution weights and the higher value-number means more related.

  

4.2.2.5 Word Embeddings

In this semantic approach three different models needed. Firstly, to be able to get semantic relations; pre-trained word embeddings are needed. Word2Vec, fasttext, GloVe are one of the alternatives. In this study pre-trained word2vec model for Turkish trained via Turkish Wikipedia articles by Abdullatif Köksal [23] is used. Since in the document similarity part of the project also uses word2vec, to use memory efficiently same word embedding is preferred. Under favour of word2vec, it is possible to represent each document as a list of vectors. To represent document as a list of vectors: mean value of each words’ vector representation is taken and listed. Secondly, k-means algorithm which is imported from sklearn is used for clustering these list of vectors. Topic number taken from user is used to fill n_clusters parameter of k-means. That means clustering -to n- vectors which actually includes semantic meaning. Lastly, after clustering process a conventional topic modeling algorithm is needed to infer topic-word distributions. In this study LDA is preferred among other conventional topic modeling algorithms because it’s time/success is quite low. LDA is applied to infer one topic to each cluster had been created.

  

4.2.3 Visualization

Since this study focuses more on functionality, visualization or discovery tools by visual means (histograms, graphics, etc.) are not much emphasized. however, 2 types of visualizations, which are thought to be helpful in understanding the proximity of the clustered documents and the relationship of the topics, were added. As an approach, these two types of visualization are identical, but one offers the possibility of 2-dimensional exploration while one offers the possibility of 3-dimensional exploration. In some cases it may be more comfortable to explore on the 3D plane, especially when many topics are created.

  

Dimensionality reduction algorithms are the fit for such these graph purposes. T-SNE and PCA are used in this study. In future selecting the dimension reduction algorithm might be an option for the user. T-SNE (t-distributed stochastic neighbor embedding) is a technique which is widely used and well suited for visualization of high dimensional datasets created by Laurens van der Maaten [25]. Document-topic distributions are created with every topic modeling algorithm used, with using these topic weights-percentages it is possible to obtain 2 dimensional values per each document. However there is a restriction which is that topic number should be selected equal or above two to create 2d graph and three or above topic number to create 3d graph. This is because it is not possible to reduce 1 dimension -one topic- to 2 dimension or 2 dimension -two topics- to 3 dimension. After getting representation of 2 dimension values which is located as x and y in coordinates, graph is made by using bokeh library. For 3d graph: dimension is reduced to three and used as x, y and z coordinates and plotly’s scatter_3d is used. Each document is represented as dots in coordinates with hover informations such as: document name, topic name document is belong to and most frequent three words used in that document.

  

4.2.4 Evaluation of Models

Even though it is possible to see the topic-word distributions and document-topic distributions and evaluate these result bu human hand, it is not practical to try different topic numbers every time and compare distributions by human evaluation. There are approaches that are applied to compare methods with each other and to compare the success of the models that are created by using different topic numbers within the same method. Perplexity measure and coherence value are the widely used evaluation criterias for topic modeling. To keep website simple only one metric is used which is coherence value. Also it is known that coherence value is studied because perplexity metric is not correlated with human evaluation. CoherenceModel is imported from gensim and it requires model, texts -tokens-, dictionary -that referred as id2word before-. Since time spent in website is important for user is important, finding optimum topic number service is available in website thanks to coherence model. Usage of this service is explained in web section. In addition, since coherence value is meaningless in word embedding approach because of topic modeling algorithm is used just for creating one topic per cluster, the measure for word embedding approach is silhouette score. This is because, actually topics are as successful as success of the clustering algorithm. Range of this score is between -1 and 1. However, silhouette score is not expected to show insights about success of model as much as coherence value does. Unfortunately, silhouette score restricts user to determine topic number. Topic-cluster number can’t be equal or higher than the number of documents given.

  

# 5. Experimental Results

# 5.1 Topic Modeling

# The kemik-Turkish news dataset includes 41,992 news and 13 titles. Since creating model for such this huge corpus requires too many computation power or time, we divided this dataset and created new sub-dataset from it and called “karma”. Karma includes 886 different news in Turkish related to titles which are world, economy, culture-art, magazine, health, politics, sport, technology and Turkey.

  

Optimum topic number (k)

Coherence value at k

Coherence Value at n

Human Evaluation

LDA

38

0.40

0.31

Incompatible

LSA

4

0.44

0.22

Incompatible

HDP

(143 topics created)

0.47

X

Incompatible

NMF

19

0.67

0.55

Coherent

Word Embeddings

X

X

Silhouette Score: 0.16

Extremely coherent

Table 4: Experimental Result of Topic Modeling Methods where n is news title (9)

  

When LDA model is created with topic number between 1 and 40, we see that the highest coherence value is obtained at topic 38 with 0.40. However when it’s topic-word and document-topic distributions are investigated we see that topics are not coherent. Since we know that “karma” consists of 9 title, when we create LDA model with 9 topics obtained coherence value is 0.31. Also it is seen that words are not coherent. However, it is obtained that we get topics of different words related to other titles. As a result, each topic contains almost the same number of documents.

  

When LSA model is created with topic number between 1 and 40, we see that the highest coherence value is obtained at topic 4 with 0.44. However when it’s topic-word and document-topic distributions are investigated we see that topics are not coherent and way worse than LDA.

  

Distributions of NMF seems accurate and words in topic, topics in document seems relatable. Therefore it is most successful method in this case among conventional topic modeling methods. However word embedding approach seems nearly perfect. Semantic relations are highly coherent, thereby distributions are victorious.

  

Conventional approaches can be considered normal in that it may not work properly with this partially short news data. For this reason, a second attempt was made with 27 books of 5 subjects, which is our second dataset. As a result of this trial, word embeddings gave the same quality result. NMF produced better results than before, and distributions were seen to be quite coherent. Even though LDA and HDP are not very compatible, they at least produced more successful results than the previous experiment. LSA was the most unsuccessful in both attempts. The inference made as a result of all these studies is quite variable depending on the quality and condition of the given data. however, it can be said that longer and more text generally results in more successful results.

  

5.2 Document Similarity

The Kemik-Turkish news dataset was used for the document similarity experimental result part. 50 news articles were selected. One news article was selected from 50 news articles and the similarity with other articles was examined by using all approaches.

The similarity between the news named 504.txt and other news has been calculated. In the tables below, the 5 most similar news are shown for all approaches.

  

This is 504 news:

  

“Ortalığı kasıp kavuracak Teknoloji devi Apple'ın iPhone'dan sonraki en büyük hamlesi olması beklenen iWatch marka akıllı kol saatleri nasıl olacak? Muhtemel hangi özellikleri taşıyacak? Bugün için pek yaygın olmasa da hemen her teknoloji şirketinin hemfikir olduğu bir konu var: 'Giyilebilir teknolojiler geleceğin trendi olacak'. Bu bağlamda Cookoo Watch ve Pebble gibi şirketler bugünden geliştirdikleri akıllı saatlerle kullanıcılarına bir saatin ötesinde yeni bir deneyim yaşatıyor. Bu iki firma dışında akıllı saatle başka kimler ilgileniyor? İnternet devi Google'a yakın bir kaynağa göre şirket halihazırda bir akıllı saat üzerinde çalışıyor. Zaten akıllı gözlük ile giyilebilir teknoloji alanında radikal bir adım atan Google'ın böylesi bir kol saatini geliştirmesi duyanları pek de şaşırtmıyor.Peki Apple akıllı kol saatin neresinde? Apple'dan bugüne dek akıllı saatlerle ilgili herhangi bir açıklama gelmedi. Ancak şirketin aldığı bir takım patentlere bakıldığında Apple'ın akıllı saatlerde kullanması muhtemel bazı donanım ve teknolojilerin varlığından haberdar oluyoruz. Bunun dışında Apple'a yakın kaynaklar da şirketin akıllı saat işinden fazla geri duramayacağının altını çiziyor.Apple'ın kol saati geliştirmesi durumunda ortaya nasıl bir şey çıkar? Kullanıcıların hayatını nasıl değiştirir?Business Insider'ın haberine göre Apple'ın akıllı saatlerinin ismi kuvvetle muhtemel iWatch ismini taşıyacak. Bunun dışında gelişmiş özellikleri olduğu için standart bir saate göre güç sorunu yaşayacağından kablosuz şarj imkanı da bulunacak. Böylece kullanıcılar telefonlarının aksine saatlerini sık sık kollarından çıkarıp şarj etmek durumunda kalmayacak.Bunun dışında akıllı saatiniz bedeninizi de yakından inceleyecek. Örneğin yürüme hızınızı, kat ettiğiniz mesafeti, yaktığınız kaloriyi ve tansiyonunuzu size an ve an bildirebilecek. NFC (Yakın Alan İletişim) teknolojisi sayesinde kredi kartı olmaksızın alışverişlerde ödeme yapabilmenize de imkan verecek olan iWatch ile film izleyip müzik dinleyebileceksiniz.Son olarak Siri uygulamasının da iWatch'ta yer alacağını düşündüğümüzde sesli komutlarla saatimizi kullanabileceğiz. Tüm bunlar elbette sadece birer iddia ve olasılıklardan ibaret... Ancak Apple'ın iWatch ile yakın gelecekte karşımıza çıkma ihtimali son derece yüksek...”

  
  

TFIDF Cosine

Similariy

TFIDF Euclidean

Distance

TFIDF Manhattan

Distance

File Name of News

Similarity

File Name of News

Distance

File Name of News

Distance

532.txt

0.4957

532.txt

0.1119

532.txt

1.417

309.txt

0.1909

52.txt

0.1198

309.txt

2.0864

39.txt

0.056

51.txt

0.1270

227.txt

2.208

464.txt

0.0455

292.txt

0.1279

322.txt

2.2347

62.txt

0.0365

222.txt

0.1285

337.txt

2.2347

Table 5 : 5 most similar news are shown for TF-IDF approaches

  

Word2Vec Cosine

Similarity

Word2Vec Euclidean

Distance

Word2Vec Manhattan

Distance

File Name of News

Similarity

File Name of News

Distance

File Name of News

Distance

532.txt

0.9479

532.txt

1.6313

532.txt

26.2482

309.txt

0.9056

309.txt

2.1550

309.txt

34.7164

330.txt

0.8942

330.txt

2.3511

330.txt

37.5661

464.txt

0.8937

464.txt

2.3916

464.txt

37.6584

327.txt

0.8632

292.txt

2.5199

292.txt

39.9976

Table 6: 5 most similar news are shown for Word2Vec approaches

  

532.txt:

  

”Apple'ın 100 kişilik ekibi akıllı saat iWatch üzerinde çalışmalarını yoğunlaştırdı.Bloomberg'ün haberine göre Apple'da seçilmiş 100 kişilik bir ekip, test aşamasını dahi geride bırakan ve muhtemelen iWatch ismini taşıyacak olan akıllı saatler üzerinde çalışıyor.Her ne kadar halihazırda bulunan saat ebatlarındaki iPod nano'lar kullanıcılara yabancı gelmese de iWatch iPod nano'nun çok daha gelişmiş bir sürümü olacak. Apple'ın üstünde özenle durduğu bu saatler özellikle Google'ın akıllı gözlüğüyle alevlendirdiği giyilebilir teknoloji pazarında önemli bir silahı olacak.iWATCH NASIL OLACAK Gelişmiş özellikleri olduğu için standart bir saate göre güç sorunu yaşayacağından saatin kablosuz şarj imkanı da bulunacak. Böylece kullanıcılar telefonlarının aksine saatlerini sık sık kollarından çıkarıp şarj etmek durumunda kalmayacak.Bunun dışında akıllı saatiniz bedeninizi de yakından inceleyecek. Örneğin yürüme hızınızı, kat ettiğiniz mesafeti, yaktığınız kaloriyi ve tansiyonunuzu size an ve an bildirebilecek. NFC (Yakın Alan İletişim) teknolojisi sayesinde kredi kartı olmaksızın alışverişlerde ödeme yapabilmenize de imkan verecek olan iWatch ile film izleyip müzik dinleyebileceksiniz.Son olarak Siri uygulamasının da iWatch'ta yer alacağını düşündüğümüzde sesli komutlarla saatimizi kullanabileceğiz.”

  

309.txt:

  

“iPhone'u kendi yok edecek Teknoloji devi Apple iPhone'un yerini alacak akıllı saat için çalışmalara başladı.

iWatch olarak da isimlendirilebilecek Apple'ın akıllı saati özellikle yakın gelecekte daha da önem kazanacak giyilebilir teknolojik aygıt pazarında önemli bir kozu olacak. Apple bu şekilde 100 dolara hemen herkese hem telefon hem saat hem müzik çalar olan bir aygıtı sunabilecek.iPad ile birlikte tabletler geleneksel bilgisayarların yerini hızla alırken Google'ın geliştirdiği akıllı gözlüğün yakın gelecekte akıllı telefonları elemine etme ihtimali söz konusu.Bu durumun farkında olan Apple ise kendi giyilebilir teknolojisini üretip iWatch ile iPhone'u ortadan kaldırmayı planlıyor. Bu sayede şirket geleceğin trendini yakalamayı başarmış olacak.Piper Jaffray'den Gene Munster konuyla ilgili yaptığı açıklamada giyilebilir teknolojilerin 10 yıl içinde yaygınlaşmasını beklediğini kaydederken, Apple'ın iPhone'u bırakıp iWatch tipi ürünlere yönelmesinin de yine bu süre zarfında olacağını savundu.”

  
  
  

In the results of all similarity approaches, the most similar news is 532.txt. The two news relate to Apple's iwatch product and features. They appear to be quite similar. The second most similar news is 309.txt. This news is about apple iwatch but not includes features. We can say that it contains similar subject with 504.txt.

  

It seems that the approaches associated with Word2vec show similar results, but the approaches formed with tf-idf show different results. When the top 5 news similarities found with Tf-IDF euclidean distance are examined, it does not give a good result compared to others.

  

# 6. Web Platform

A web application has been developed to make all outputs made in this study open and available to everyone. Since the vast majority of NLP libraries are written in Python, Django, a library written in python, was used for the web interface. Django uses MVT as a software design pattern. MVT (Model View Template) is a collection that includes three important components. The Model helps to handle database. It is a data access layer to handle data. The Template is a presentation layer for handling User Interface it usually includes HTML files. The view is kind of bridge between Model and Template, it collects data, executes business logic and returns rendered template.

  

In the Django Framework each module named as application, in the project there are 3 applications, Project, Topic Modelling and Document Similarity. Project, handles creating projects and uploading files, Topic Modelling and Document Similarity, handles applying algorithms to project, creates, stores and render report pages.

  

Projects and Reports are stored in SQLite database which comes with Django Framework and Project Files are stored in file system. Front-end of the web application done with HTML5, Bootstrap, JQuery libraries.

  

![](https://lh5.googleusercontent.com/ZUAGcm373WqKQmCE_Qjm2YFfBInTBY5f5bkDoooweZYNnvO6mfHkh1NIrFea1Bc7paaDQ85IVJY8Hru9ueEsv92WKmIEjyJLlbaRarcwnx040DPZOyI56HoqchWEDh_6V0iDqzCs)

Figure 2: Database Schema in UML format

## 6.1 Workflow

The user enters Projects page to see the projects or create a new project. Project includes the title and the files section, files can be .pdf or .txt format. After uploading files, user returns Projects page.

  

If user clicks any project, the next page shows two options to apply: Topic Modelling and Document Similarity.

  

-   If user selects Topic Modelling, the next page shows 5 different topic modelling algorithms to apply: LDA, LSA, HDP, NMF and Word2Vec LDA. After selecting one of these algorithms user sets the topic count (except for HDP) and runs the algorithm, after algorithm ends user returns to the Report page which shows the outputs of the algorithm and the final part of workflow.
    

-   If user selects Document Similarity, the next page shows 6 different document similarity algorithms to apply: TFIDF-Cosine Similarity, TFIDF-Euclidean Distance, TFIDF-Manhattan Distance, Word2Vec-Cosine Similarity, Word2Vec-Euclidean Distance and Word2Vec-Manhattan Distance. After selecting one of these algorithms user selects a document to compare with the others documents in the same project then runs the algorithm, after algorithm ends user returns to the Report page which shows the outputs of the algorithm and the final part of workflow.
    

  

# 7. Future Works

# It is understood from this study that when semantic relationships are caught, the relationships between texts can be detected much more successfully. In this study, although trying to establish semantic patterns using word2vec, alternative word embeddings have not been tested. Their performance should be compared using alternatives such as Fasttext and GloVe. Besides, even if it is not appropriate to integrate it into the web application, the conceptual word embeddings approach should be tried with the word-vector approaches that do not ignore the different values that vectors can gain in the concept. The concept is ignored in the traditional word2vec approach. More successful inferences can be made using models such as BERT.

In the document similarity part of the study, doc2vec method can be tried and knowledge graph approaches can be compared.

Since this study focused on functionality rather than visuality, the visuality was weak. With the graph and histogram etc., which enables the documents to be discovered better, the user can be helped to know the data more closely.

In the current web application, every process is running in the main thread. If a queuing system such as Redis is used, the user can do long operations without any problems.

  
  

8. Conclusion

  

Automation of information processing takes an important place in today's data age. It is possible to successfully extract meaning from articles with mathematical, statistical and data-based artificial intelligence / natural language processes. In this study, a web application was developed in which people who do not have knowledge of programming can also use the options of document-similarity and topic-modeling tools to make use of natural language processing techniques. Thanks to the product produced as a result of this study, conventional and modern approaches for topic-modeling and document similarity operations can be easily and quickly tried and compared. All the work has been developed as an open source [26] and is available to humanity free of charge.

REFERENCES

[1] Prem K Gopalan, Laurent Charlin, and David Blei. 2014. Content-based recommendations with poisson factorization. In Advances in Neural InformationProcessingSystems,pages3176–3184

[2] Jorg Ontrup and Helge Ritter. 2002. Hyperbolic selforganizing maps for semantic navigation. In Advances in neural information processing systems, pages1417–1424.

[3] Thomas K Landauer. 2003. Automatic essay assessment. Assessment in education: Principles, policy &practice,10(3):295–308

  

[4] Mamdouh Farouk “Measuring Sentences Similarity: A Survey “ Indian Journal of Science and Technology,

[5] Gomaa, Wael H., and Aly A. Fahmy. “A survey of text similarity approaches.”International Journal of Computer Applications

[6] Aditi Gupta , Mr. Ajay Kumar, Dr. Jyoti Gautam “A Survey on Semantic Similarity Measures ” International Journal for Innovative Research in Science & Technology

[7] Didik Dwi Prasetya,Aji Prasetya Wibawa, Tsukasa Hirashima “The performance of text similarity algorithms” International Journal of Advances in İntelligent İnformatics

[8] P.Sitikhu, K.Pahi, P.Thapa, S.Shakya “A Comparison of Semantic Similarity Methods for Maximum Human Interpretability” IEEE International Conference on Artificial Intelligence for Transforming Business and Society, 2019

[9] Erwan Moreau, François Yvon, Olivier Cappé. Robust similarity measures for named entities matching. COLING 2008, Aug 2008, Manchester, United Kingdom. pp.593-600. hal-00487084

[10] Radha mothukuri, Nagaraju.M, Divya Chilukuri “SIMILARITY MEASURE FOR TEXT CLASSIFICATION ” International Journal of Emerging Trends and Technology in computer science

[11] Fiona Martin, Mark Johnson. 2015. More Efficient Topic Modelling Through a Noun Only Approach

[12] Fabrizio Esposito, Anno Corazza and Francesco Cutugno. 2015. Topic Modeling with Word Embeddings

[13] Pooja Kherwa and Poonam Bansal. 2019. Topic Modeling: A Comprehensive Review

[14] Suhyeon Kim, Haecheong Park, Junghye Lee. 2020. Word2vec-based latent semantic analysis (W2V-LSA) for topic modeling: A study on blockchain technology trend analysis

[15] [http://kemik.yildiz.edu.tr/](http://kemik.yildiz.edu.tr/)

[16] M.K.Vijaymeena and K.Kavitha(2016). “A survey on Similarity Measures in Text Mining”

  

[17] Tomas Mikolov, Kai Chen, Greg Corrado, Jeffrey Dean. 2013. “Efficient Estimation of Word Representations in Vector Space”

  

[18]David M. Blei,Andrew Y. Ng, Michael I. Jordan. 2003. “Latent Dirichlet Allocation”

[19] Zhi-Yong Shen,Z.Y., Sun,J., and Yi-Dong Shen,Y.D., ―Collective Latent Dirichlet Allocationǁ, Eighth IEEE International Conference on Data Mining, pages 1019–1025, 2008.

[20] X. Wang and A. McCallum. ―Topics over time: a non-markov continuous-time model of topical trendsǁ. In International conference on Knowledge discovery and data mining,

[21] Chong Wang, John Paisley and David M. Blei. 2011. Online Variational Inference for the Hierarchical Dirichlet Process.

[22]Chenliang Li1, Haoran Wang1, Zhiqian Zhang1, Aixin Sun2, Zongyang. 2011 .Ma2Topic Modeling for Short Texts with Auxiliary Word Embeddings.

[23][https://github.com/akoksal/Turkish-Word2Vec](https://github.com/akoksal/Turkish-Word2Vec)

[24] Fiona Martin,Mark Johnson. 2015 .More Efficient Topic Modelling Through a Noun Only Approach

[25] [https://lvdmaaten.github.io/tsne/](https://lvdmaaten.github.io/tsne/)

[26] https://github.com/BilgiAILAB

  

# APPENDIX

Web Application Interfaces

Home

![](https://lh5.googleusercontent.com/kTqvrfw3QIZmVjLfhv8BYXtde4h2V3SfUXks9dScv8WPPF92XmK0p373GjhRViuUEQsy3DYRe8qGOcatjtnyyHM__evUhh1j8_8ha9FGYCKXGtltp4w32E0jkVa3aXl4jjkrjv82)

All Projects

![](https://lh3.googleusercontent.com/Lta5CGvwHvYMVq0ibNCBHemUDHWk544N9MpCrVzYc0ZXOQozn51pemL-o8DqbT7dfHj_Przm_ALpkDqaCXFY-Pdp7cq2e4tJmC_aLzIZe-l838C9ZVfm4_3HzALxuvMN79tH6P7v)

  
  
  
  
  

Create Project

![](https://lh4.googleusercontent.com/t3juyW5ezhM35SCYQ6rbDlw0tz3xGO9dPy55zMKX05VkquHVGUWOqeVB0Rl_1ob7S3PZZ970slyUf_sgcfRE5jOobNvy1MG5VZKnIbR20abG0P_8BLHzGTfxX2I-55HHTrGxyRJ8)

  
  
  
  
  
  

Show Project

![](https://lh6.googleusercontent.com/ycTOfPyEaLm4Mked8BuhywYb0nMxsw7UKNF8YThfcnxAl7JNxKb9sjp8-wPllUowvoxk6w_qett9R4RikzCXwHfhcOH0ZTdEDUCju7H_jYEQrm72p1cqft4kjADZMp1cnBVbe5Jr)

  
  

Show Project > Document Similarity

![](https://lh4.googleusercontent.com/-qkpC5WHvQUqHF0tNEKg_0IJyYcpF8Ap065I8ZruqSnZUVgb4DxAN9jMOasptf9RLwYB8aFCsI4y1aZQ8_51cAtugeIGDoXxd_FBIdPtgsjtz-YJqre7ks_-iQ7MbBVALjG4UIGS)

  
  
  
  
  
  
  
  

Show Project > Document Similarity > Apply TFIDF-COS

![](https://lh6.googleusercontent.com/tbjtBims-BPJpcr3Czrtn3wt9p9Uk48WOXEblbVBaZjnWkszU08Lc16C2KtjQuwpI8eu3eL3nL9nYYZigx-Y_qCELb2nXku88t2LtpqZwAufrKSULwnxw285sfLLW-PcMILIM66a)

Show Project > Document Similarity > Apply TFIDF-COS > Report

![](https://lh4.googleusercontent.com/Xbr2D-ZG7AaYHU903LndwLbin-OiFr5aNrYaXePUFlnaCfn1gsUTlvSPJ3cHXcjn6KZ_jjlnuJdpna-hjIbcWCUhAnpkVICYD3PvdhfD77BiMwUkXKfTZ0rhvrkPiRvpPJYtWzwE)

  
  
  
  
  
  
  
  

Show Project > Topic Modeling

![](https://lh5.googleusercontent.com/Ex7qiRvCg80YmvafxT1An2OmnnAJcUsr3I7E4MuTmQRp0uJN7JmRrmbVu2BuIaGPcYR1tK6Gltx-4fkk0kTKbfxKjXNSgAmC8Ow_HCrUvBbQVU_Ucrg0hSBf6rQ-P4wFKf2zOqCT)

Show Project > Topic Modeling > Apply LDA

![](https://lh5.googleusercontent.com/FxyD1QKAhfdJBb-WwjYZKU4m4gk7ZKG6tK7RRfw98Zta5S2JWcykbeJ_AR0wpmiX242rq-_y1JeBDbgUC60wpLJfiaMTwhLh3E6T7xpQhezBYpziYJjFusLpv5be7dVoCQ36-Ci8)

  
  
  
  
  
  

Show Project > Topic Modeling > Apply NMF > Step

  

![](https://lh5.googleusercontent.com/yNSB5EZ1rgddIzunZ_myE5jGe_AmdThJOE6GfLVh51A9S71asF8MQ4m70DLFPRFQz6QdTsnl89Zlbz6-QpkWMOrFA6PJYTsy4gNZi7pIgd6naaZRdEz2WKB99MdM_rU07BauvlyI)

  

Show Project > Topic Modeling > Apply LDA > Report

![](https://lh3.googleusercontent.com/epQtpKdq3JB-ZLc10uIuIInRJvl6XqM76IsExaZeoJFv35L_pOEEj4W9wZQGjh3t_-m3BUPjWlkTGALSD_DHi72tYbxJBbhIYhGsXD15n--s8cms7hCw-bH18XWdxG2xR2iuxMG-)

  
  
  
  
  
  

Show Project > Topic Modeling > Apply LDA > Report - 2D TSNE Graph

![](https://lh3.googleusercontent.com/ft8kbkLy518iHUvGiSDrUa8YuNHgapIa2jmRVM3JAqPKst4kcjJm9Rivs5oBcxqADUnT7Y42uAYDhtJu2hN3n8w7Q-h1QiFhjXEXhPYdlb6_YZRghBqLcXBVboGcvKxgfDRCjO2v)

  
  
  

Show Project > Topic Modeling > Apply LDA > Report - 3D TSNE Graph

![](https://lh4.googleusercontent.com/vPb30rwsPo5MZ90RMq5aIHs8DqFRADbwEgXpO7SEAvQ8EadpY9Z-1wc48nGnidrRpgInSpBXMUkAZclGwtpkHh7BroKiEEuCwrXvA_46oVpCVSX5SI7e7Qn7RBbf6TYo4auXCe-y)
