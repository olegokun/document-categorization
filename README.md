![Project Logo](https://cdn.pixabay.com/photo/2017/07/15/22/07/library-2507902__340.jpg)
# Document Categorization
Automatic partitioning the collection of my e-books into categories and labeling each category according to its content

## Problem formulation: Unsupervised document clustering + topic modeling
The goal of this project was to automatically organize my large collection of e-books in the PDF format into groups or clusters so that I could easily find sources of useful information when needed. However, the code in this repo could be handful for the general purpose where there are many PDF files that one wants to split into groups where each group contains books or documents on a similar subject/topic. In machine learning terms, such a task is named document categorization and it is completely **unsupervised**.

More description will be added soon!

## Data
As an example, I used 21 e-books from my personal collection (due to the copyrights I cannot upload these books here). Here is their list:
* Advanced Deep Learning with Keras
* Advanced Deep Learning with Python
* etc.

These books could roughly be divided into 3-4 clusters.

## Running code
The main file with all necessary code to execute in your favorite IDE or from the command line is *document_categorization.py*. The file *categorization.env* is the environment file where all important parameters, such as clustering method or the number of topics per cluster, are set up. I used a lot of code from the great book ["Text Analytics with Python: A Practical Real-World Approach to Gaining Actionable Insights from Your Data"](https://www.apress.com/gp/book/9781484223888), written by *Dipanjan Sarkar* and published by Apress in 2016. My role was to write so called integration code linking together different parts of the processing pipeline described in the next section. Whenever the code has been adopted, I preserved the original file and function names given by Dipanjan Sarkar.

## Processing pipeline
Text filtering -> Document clustering -> Topic modeling

## Results
There are three clustering methods ([affinity propagation](https://en.wikipedia.org/wiki/Affinity_propagation), [k-means](https://en.wikipedia.org/wiki/K-means_clustering) and [Ward's hierarchical clustering](https://en.wikipedia.org/wiki/Ward%27s_method)) and two topic modeling methods ([Latent Dirichlet Allocation or LDA](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation) and [Nonnegative Matrix Factorization or NMF](https://en.wikipedia.org/wiki/Non-negative_matrix_factorization)).

[Affinity propagation](https://en.wikipedia.org/wiki/Affinity_propagation) does not require to pre-specify the number of clusters to be found in advance, unlike [k-means](https://en.wikipedia.org/wiki/K-means_clustering). Although [Ward's hierarchical clustering](https://en.wikipedia.org/wiki/Ward%27s_method) also does not need to know that number in advance, this clustering method requires a human to judge on the final number of clusters from a dendrogram, i.e., clustering partitioning is rather subjective. Having decided on this number, a user can then supply it to [k-means](https://en.wikipedia.org/wiki/K-means_clustering). The dendrogram for my set of 21 books is shown below.

![dendrogram](https://github.com/olegokun/document-categorization/blob/master/ward_hierachical_clusters.png)

One could observe 3 clusters presented by red, green and light blue lines. One cluster includes all books about Deep Learning, another one about Apache Spark, and the third one about search engines/platforms (Elasticsearch and Solr). 

Given these considerations, I decided to go with [affinity propagation](https://en.wikipedia.org/wiki/Affinity_propagation), as it is rearly that the number of clusters is known beforehand. All results below are obtained with this clustering method.

Word clouds for each of the extracted clusters are given below.

![Wordcloud for Cluster_0](https://github.com/olegokun/document-categorization/blob/master/cluster_0.png)

![Wordcloud for Cluster_1](https://github.com/olegokun/document-categorization/blob/master/cluster_1.png)

![Wordcloud for Cluster_2](https://github.com/olegokun/document-categorization/blob/master/cluster_2.png)

![Wordcloud for Cluster_3](https://github.com/olegokun/document-categorization/blob/master/cluster_3.png)

![Affinity propagation results](https://github.com/olegokun/document-categorization/blob/master/clustering_results.png)

A book title and its content from extracted top bigrams and trigrams are written to an SQLite database (file *documents.sqlite*). However, there is a check preventing any book to be written more than once in order to avoid duplicated records and unnecessary database growth.

## Potential future improvements
I observed that tokens from a programming code sometimes polluted clusters. This happened because many of my books contain a lot of code snippets and text pre-processing, despite being rigorous, was unable to clean up these artifacts. One potential solution of this problem could be paragraph extraction, e.g., based on some heuristics such as blank lines between paragraphs, followed by paragraph classification into code and plain text. Naturally, the latter would require a one-class or binary classifier trained on examples of code in several popular programming languages and, if a binary classifier is used, plain text. The goal is to filter out paragraphs (almost) entirely consisting of code, while leaving paragraphs with a minor fraction of code untouched as few instances of code in the whole large book would unlikely result in the high [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) score and hence, such "noisy" tokens won't do much harm to document clustering.
