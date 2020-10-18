# Welcome to Bilgi AILab

---

This website has been prepared as a thesis project by Bilgi University Computer Engineering students İbrahim Doğan, Hanefi Enes Gül and Simge Erek. We would like to thank our thesis advisor Tuğba Yıldız teacher.

In this project, Django, Bootstrap 4, Bokeh, Plot.ly, Ionicons tools were used. Thanks also to the open source developers.

Data is stored publicly on Digital Ocean servers, do not upload your sensitive data.



# Create a Project

---

What we call the project actually refers to the collection of documents. You can touch the + icon on the Projects tab to create a project. When creating a project, just enter your project name and add your documents. Each document should be found as a file.

We currently only support .txt and .pdf file formats.



#### Add New Files To Project

---

Go to the Projects tab and click on the project you want to add new files, press the Add New Files button.



#### Remove Project

---

Go to the Projects tab and click on the project you want to delete and click the delete project button on the page that opens.



# Select Algorithm

---

Select a project from the Projects tab, press Apply Document Similarity and choose one of the algorithms you can apply:


1. TFIDF-Cosine Similarity

2. TFIDF-Euclidean Distance

3. TFIDF-Manhattan Distance

4. Word2Vec-Cosine Similarity

5. Word2Vec-Euclidean Distance

6. Word2Vec-Manhattan Distance



# Apply Algorithm

---

After selecting one of the Document Similarity algorithms, you should select a document from the project to compare with other files in the same project and click Run button.



# Explore Results

---

After applying the algorithm, they are not lost and saved as a report, you do not need to re-apply them when you want to look at them. You can click History and click View.

It comes with rank of loading files that contain similarities rate table as default, if you want to rank in the table according to their similarity Similarity / Distance Click on the title.

If you want to download the files, click the Download Files button,
you can select all the files or filter them by a certain percentage and download them.



# Select Algorithm

---

Select a project from the Projects tab, press Apply Topic Modeling and choose one of the algorithms you can apply:

1. LDA

2. LSA

3. HDP

4. NMF

5. Word2Vec Kmeans



# Apply Algorithm

---

After selecting one of the Topic Modeling algorithms, enter number of topics to create and click Run button. (except HDP algorithm, it does not require topic number)



To find optimum number of topic count, you can use Step section. Put start, end and step numbers to run and create coherence value graph for each topic number.

Example:

Start: 2 , End: 10, Step: 1 will calculate for 2, 3, 4, 5, 6, 7, 8, 9 topics and returns their coherence value as graph. Since it is running for multiple times, it can take a while.



# Explore Results

---

After applying the algorithm, they are not lost and saved as a report, you do not need to re-apply them when you want to look at them. You can check History and click View.



There is 3 section in Topic Modelin report.

1. Topic Word Distrubution Table

By default:

- Table is sorted by Topic Number, you can click the table header to sort them.

- Topic names are from Topic 1 to Topic N, you can change it by clicking Edit Topic Names, after you rename it click Apply button.

- In words column 10 keywords are showing respect to their importance, you can change it from 1 to 20, to do that click Filter and use slider.



You can preview and download files by clicking Count of Documents. When you click Download files button, it will download all of the documents related with that topic.



2. Document Topic Distrubution

By selecting a document and clicking Add button, you can see topic distrubution of that document. If you want to add all documents at once you can click Show All button.

3. TSNE Graphs

You can visualize document topic distrubutions by TSNE graphs 2D or 3D. To see the graphs click Load Graph button.



# Contrubuting

We love and support open source, if you want to contribute, you can send us your ideas or send a pull request via github.

Github Repository: https://github.com/ibrahim-dogan/cmpe-graduation-project
