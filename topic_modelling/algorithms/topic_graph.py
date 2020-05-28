from collections import Counter
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
from bokeh.embed import file_html
from bokeh.models import HoverTool, ColumnDataSource
from bokeh.plotting import figure
from bokeh.resources import CDN
from sklearn.manifold import TSNE


def kmeans_3d_graph(doc_vectors, topic_names, doc_names, colorArrayy, labels, data_tokens):
    freq_per_doc = []
    for document in data_tokens:
        most_vocab = Counter()
        for word in document:
            most_vocab[word] += 1
        freq_per_doc.append([word_tuple[0] for word_tuple in most_vocab.most_common(3)])

    pca = PCA(n_components=3).fit(doc_vectors)
    datapoint = pca.transform(doc_vectors)
    df = pd.DataFrame()
    df['x'] = datapoint[:, 0]
    df['y'] = datapoint[:, 1]
    df['z'] = datapoint[:, 2]
    df['topic_name'] = np.array(topic_names)[labels]
    df['file_name'] = doc_names
    df['frequent_words'] = freq_per_doc
    fig = px.scatter_3d(df,
                        x='x',
                        y='y',
                        z='z',
                        color='topic_name',
                        opacity=0.8,
                        size_max=18,
                        hover_data=['topic_name', 'file_name', 'frequent_words'],
                        )

    html_plot = plotly.offline.plot(fig, output_type='div')
    return html_plot

def tsne_graph_3d(output, topic_names, doc_names, algorithm):
    colorArray = [
        "#63b598", "#ce7d78", "#ea9e70", "#0d5ac1",
        "#14a9ad", "#4ca2f9",
        "#d298e2", "#6119d0", "#d2737d", "#c0a43c", "#f2510e",
        "#651be6", "#79806e", "#cd2f00", "#9348af",
        "#c5a4fb", "#996635", "#b11573", "#4bb473", "#75d89e",
        "#2f3f94", "#2f7b99", "#da967d", "#34891f", "#ca4751",
        "#4b5bdc", "#250662", "#cb5bea", "#228916",
        "#ac3e1b", "#df514a", "#539397", "#880977", "#f697c1",
        "#f158bf", "#e145ba", "#ee91e3", "#4834d0",
        "#802234", "#6749e8", "#8fb413", "#b2b4f0",
        "#c3c89d", "#c9a941", "#41d158", "#fb21a3", "#51aed9",
        "#5bb32d", "#0807fb", "#21538e", "#89d534", "#d36647",
        "#7fb411", "#0023b8", "#3b8c2a", "#986b53", "#f50422",
        "#983f7a", "#ea24a3", "#79352c", "#521250", "#c79ed2",
        "#63b598", "#ce7d78", "#ea9e70", "#0d5ac1",
        "#14a9ad", "#4ca2f9",
        "#d298e2", "#6119d0", "#d2737d", "#c0a43c", "#f2510e",
        "#651be6", "#79806e", "#cd2f00", "#9348af",
        "#c5a4fb", "#996635", "#b11573", "#4bb473", "#75d89e",
        "#2f3f94", "#2f7b99", "#da967d", "#34891f", "#ca4751",
        "#4b5bdc", "#250662", "#cb5bea", "#228916",
        "#ac3e1b", "#df514a", "#539397", "#880977", "#f697c1",
        "#f158bf", "#e145ba", "#ee91e3", "#4834d0",
        "#802234", "#6749e8", "#8fb413", "#b2b4f0",
        "#c3c89d", "#c9a941", "#41d158", "#fb21a3", "#51aed9",
        "#5bb32d", "#0807fb", "#21538e", "#89d534", "#d36647",
        "#7fb411", "#0023b8", "#3b8c2a", "#986b53", "#f50422",
        "#983f7a", "#ea24a3", "#79352c", "#521250", "#c79ed2",
        "#63b598", "#ce7d78", "#ea9e70", "#0d5ac1",
        "#14a9ad", "#4ca2f9",
        "#d298e2", "#6119d0", "#d2737d", "#c0a43c", "#f2510e",
        "#651be6", "#79806e", "#cd2f00", "#9348af",
        "#c5a4fb", "#996635", "#b11573", "#4bb473", "#75d89e",
        "#2f3f94", "#2f7b99", "#da967d", "#34891f", "#ca4751",
        "#4b5bdc", "#250662", "#cb5bea", "#228916",
        "#ac3e1b", "#df514a", "#539397", "#880977", "#f697c1",
        "#f158bf", "#e145ba", "#ee91e3", "#4834d0",
        "#802234", "#6749e8", "#8fb413", "#b2b4f0",
        "#c3c89d", "#c9a941", "#41d158", "#fb21a3", "#51aed9",
        "#5bb32d", "#0807fb", "#21538e", "#89d534", "#d36647",
        "#7fb411", "#0023b8", "#3b8c2a", "#986b53", "#f50422",
        "#983f7a", "#ea24a3", "#79352c", "#521250", "#c79ed2"]
    colorArrayy = np.array(colorArray)
    if algorithm.lower() == "w2v-kmeans":
        return kmeans_3d_graph(output.get("doc_vectors"), topic_names, doc_names, colorArrayy, output.get("labels"), output.get("data_tokens"))

    n_topics = len(topic_names)
    topic_distributions = output.get('topic_distributions')
    data_tokens = output.get('data_tokens')

    if algorithm.lower() == "hdp":
        topic_distributions = fill_with_zero(n_topics, topic_distributions)


    freq_per_doc = []
    for document in data_tokens:
        most_vocab = Counter()
        for word in document:
            most_vocab[word] += 1
        freq_per_doc.append([word_tuple[0] for word_tuple in most_vocab.most_common(3)])

    topic_weights = []
    for document in topic_distributions:
        topic_weights.append([abs(float(topic[1])) for topic in document])

    topic_num = np.argmax(topic_weights, axis=1)
    tsne_model = TSNE(n_components=3, verbose=1, random_state=0, angle=.99, init='pca')
    tsne_lda = tsne_model.fit_transform(topic_weights)

    df = pd.DataFrame()
    df['x'] = tsne_lda[:, 0]
    df['y'] = tsne_lda[:, 1]
    df['z'] = tsne_lda[:, 2]
    df['topic_name'] = np.array(topic_names)[topic_num]
    df['file_name'] = doc_names
    df['frequent_words'] = freq_per_doc
    fig = px.scatter_3d(df,
                        x='x',
                        y='y',
                        z='z',
                        color='topic_name',
                        opacity=0.8,
                        size_max=18,
                        hover_data=['topic_name', 'file_name', 'frequent_words'],
                        )

    html_plot = plotly.offline.plot(fig, output_type='div')
    return html_plot


def kmeans_2d_graph(doc_vectors, topic_names, doc_names, colorArrayy, labels, data_tokens):
    pca = PCA(n_components=2).fit(doc_vectors)
    datapoint = pca.transform(doc_vectors)
    labelss = np.array(topic_names)
    freq_per_doc = []
    for document in data_tokens:
        most_vocab = Counter()
        for word in document:
            most_vocab[word] += 1
        freq_per_doc.append([word_tuple[0] for word_tuple in most_vocab.most_common(3)])

    source = ColumnDataSource(dict(
        x=datapoint[:, 0],
        y=datapoint[:, 1],
        color=colorArrayy[labels],
        labels=labelss[labels],
        content=doc_names,
        frequent_words=freq_per_doc,
        topic_no=labels
    ))
    TOOLTIPS = [
        ("File Name", "@content"),
        ("Keywords", "@frequent_words"),
        ("Topic Name", "@labels")
    ]

    plot = figure(title="PCA Clustering of {} Topics".format(len(topic_names)),
                  plot_width=900,
                  plot_height=700,
                  tools="pan,wheel_zoom,box_zoom,reset,hover",
                  min_border=1,
                  tooltips=TOOLTIPS
                  )

    r = plot.scatter(x='x',
                     y='y',
                     source=source,
                     color='color',
                     legend_field='labels',
                     alpha=0.9,
                     size=10
                     )

    hover = plot.select(dict(type=HoverTool))
    # hover.tooltips = {"content": "Title: @title"}
    plot.legend.location = "top_left"
    return file_html(plot, CDN, "tsne_lda_graph")


def tsne_graph_2d(output, topic_names, doc_names, algorithm):
    colorArray = [
        "#63b598", "#ce7d78", "#ea9e70", "#0d5ac1",
        "#14a9ad", "#4ca2f9",
        "#d298e2", "#6119d0", "#d2737d", "#c0a43c", "#f2510e",
        "#651be6", "#79806e", "#cd2f00", "#9348af",
        "#c5a4fb", "#996635", "#b11573", "#4bb473", "#75d89e",
        "#2f3f94", "#2f7b99", "#da967d", "#34891f", "#ca4751",
        "#4b5bdc", "#250662", "#cb5bea", "#228916",
        "#ac3e1b", "#df514a", "#539397", "#880977", "#f697c1",
        "#f158bf", "#e145ba", "#ee91e3", "#4834d0",
        "#802234", "#6749e8", "#8fb413", "#b2b4f0",
        "#c3c89d", "#c9a941", "#41d158", "#fb21a3", "#51aed9",
        "#5bb32d", "#0807fb", "#21538e", "#89d534", "#d36647",
        "#7fb411", "#0023b8", "#3b8c2a", "#986b53", "#f50422",
        "#983f7a", "#ea24a3", "#79352c", "#521250", "#c79ed2",
        "#63b598", "#ce7d78", "#ea9e70", "#0d5ac1",
        "#14a9ad", "#4ca2f9",
        "#d298e2", "#6119d0", "#d2737d", "#c0a43c", "#f2510e",
        "#651be6", "#79806e", "#cd2f00", "#9348af",
        "#c5a4fb", "#996635", "#b11573", "#4bb473", "#75d89e",
        "#2f3f94", "#2f7b99", "#da967d", "#34891f", "#ca4751",
        "#4b5bdc", "#250662", "#cb5bea", "#228916",
        "#ac3e1b", "#df514a", "#539397", "#880977", "#f697c1",
        "#f158bf", "#e145ba", "#ee91e3", "#4834d0",
        "#802234", "#6749e8", "#8fb413", "#b2b4f0",
        "#c3c89d", "#c9a941", "#41d158", "#fb21a3", "#51aed9",
        "#5bb32d", "#0807fb", "#21538e", "#89d534", "#d36647",
        "#7fb411", "#0023b8", "#3b8c2a", "#986b53", "#f50422",
        "#983f7a", "#ea24a3", "#79352c", "#521250", "#c79ed2",
        "#63b598", "#ce7d78", "#ea9e70", "#0d5ac1",
        "#14a9ad", "#4ca2f9",
        "#d298e2", "#6119d0", "#d2737d", "#c0a43c", "#f2510e",
        "#651be6", "#79806e", "#cd2f00", "#9348af",
        "#c5a4fb", "#996635", "#b11573", "#4bb473", "#75d89e",
        "#2f3f94", "#2f7b99", "#da967d", "#34891f", "#ca4751",
        "#4b5bdc", "#250662", "#cb5bea", "#228916",
        "#ac3e1b", "#df514a", "#539397", "#880977", "#f697c1",
        "#f158bf", "#e145ba", "#ee91e3", "#4834d0",
        "#802234", "#6749e8", "#8fb413", "#b2b4f0",
        "#c3c89d", "#c9a941", "#41d158", "#fb21a3", "#51aed9",
        "#5bb32d", "#0807fb", "#21538e", "#89d534", "#d36647",
        "#7fb411", "#0023b8", "#3b8c2a", "#986b53", "#f50422",
        "#983f7a", "#ea24a3", "#79352c", "#521250", "#c79ed2"]
    colorArrayy = np.array(colorArray)
    if algorithm.lower() == "w2v-kmeans":
        return kmeans_2d_graph(output.get("doc_vectors"), topic_names, doc_names, colorArrayy, output.get("labels"), output.get("data_tokens"))

    n_topics = len(topic_names)
    topic_distributions = output.get('topic_distributions')
    data_tokens = output.get('data_tokens')

    if algorithm.lower() == "hdp":
        topic_distributions = fill_with_zero(n_topics, topic_distributions)


    freq_per_doc = []
    for document in data_tokens:
        most_vocab = Counter()
        for word in document:
            most_vocab[word] += 1
        freq_per_doc.append([word_tuple[0] for word_tuple in most_vocab.most_common(3)])

    topic_weights = []
    for document in topic_distributions:
        topic_weights.append([abs(float(topic[1])) for topic in document])

    topic_num = np.argmax(topic_weights, axis=1)
    tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99, init='pca')
    tsne_lda = tsne_model.fit_transform(topic_weights)

    labelss = np.array(topic_names)
    source = ColumnDataSource(dict(
        x=tsne_lda[:, 0],
        y=tsne_lda[:, 1],
        color=colorArrayy[topic_num],
        labels=labelss[topic_num],
        content=doc_names,
        frequent_words=freq_per_doc,
        topic_no=topic_num
    ))
    TOOLTIPS = [
        ("File Name", "@content"),
        ("Keywords", "@frequent_words"),
        ("Topic Name", "@labels")
    ]

    plot = figure(title="t-SNE Clustering of {} LDA Topics".format(n_topics),
                  plot_width=900,
                  plot_height=700,
                  tools="pan,wheel_zoom,box_zoom,reset,hover",
                  min_border=1,
                  tooltips=TOOLTIPS
                  )

    r = plot.scatter(x='x',
                     y='y',
                     source=source,
                     color='color',
                     legend_field='labels',
                     alpha=0.9,
                     size=10
                     )

    hover = plot.select(dict(type=HoverTool))
    # hover.tooltips = {"content": "Title: @title"}
    plot.legend.location = "top_left"
    return file_html(plot, CDN, "tsne_lda_graph")


def fill_with_zero(n_topics, topic_distributions):
    new_topic = []
    for document in topic_distributions:
        temp = []
        index_temp = []

        for topic in document:
            index_temp.append(topic[0])
        j_ = 0
        for i in range(n_topics):
            if i in index_temp:
                temp.append([i, document[j_][1]])
                j_ += 1
            else:
                temp.append([i, 0])
        new_topic.append(temp)
    return new_topic
