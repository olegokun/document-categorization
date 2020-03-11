![Project Logo](https://cdn.pixabay.com/photo/2017/07/15/22/07/library-2507902__340.jpg)
# Document Categorization
Automatic partitioning the collection of my e-books into categories and labeling each category according to its content

## Problem formulation: Unsupervised document clustering + topic modeling
The goal of this project was to automatically organize my large collection of e-books in the PDF format into groups or clusters so that I could easily find sources of useful information when needed. However, the code in this repo could be handful for the general purpose where there are many PDF files that one wants to split into groups where each group contains books or documents on a similar subject/topic. In machine learning terms, such a task is named document categorization and it is completely **unsupervised**.

## Data
As an example, I used 21 e-books from my personal collection (due to the copyright restrictions I cannot upload these books here). Here is their list:
* A-Gentle-Introduction-to-Apache-Spark
* Advanced Deep Learning with Keras
* Advanced Deep Learning with Python
* Advanced Elasticsearch 7.0
* Advanced_Analytics_with_Spark
* Apache Spark 2.x Machine Learning Cookbook
* Apache Spark 2.x for Java Developer
* Apache Spark Deep Learning Cookbook
* Apache_Solr_Essentials
* Apache_Solr_High_Performance
* Deep Learning with TensorFlow 2 and Keras - Second Edition
* Deep_Learning_for_Search
* Deep_Learning_with_JavaScript
* Elasticsearch 5.x Cookbook - Third Edition
* Elasticsearch 7 Quick Start Guide
* Elasticsearch A Complete Guide
* Elasticsearch_for_Hadoop
* Hands-On Deep Learning for IoT
* Learning Elastic Stack 6.0
* Learning Elasticsearch
* Mastering ElasticSearch

These books could roughly be divided into 3-4 clusters: "Spark", "Deep Learning", "Elasticsearch/Solr". I intentionally selected such books so that there would be no (significant) overlap in their topics and document clustering would have clear targets.

## Running code
The main file with all necessary code to execute in your favorite IDE or from the command line is *document_categorization.py*. The file *categorization.env* is the environment file where all important parameters, such as clustering method or the number of topics per cluster, are set up as follows:

```
# Set a directory with electronic books
BOOK_PATH="C:/eBooks/A"
# The number of top bigrams/trigrams to select
TOP_N=100
# Clustering algorithm (valid values are "affinity", "kmeans", "hierarchical")
CLUSTERING="affinity"
# The number of clusters to detect (only for CLUSTERING="kmeans")
CLUSTER_NUMBER=3
# The number of features to describe each cluster
FEATURE_NUMBER=10
# Topic modeling algorithm # (valid values are "lda", "nmf")
TOPIC_MODELING="lda"
# The number of topics per cluster
TOPIC_NUMBER_PER_CLUSTER=1
```
I used a lot of code from the great book ["Text Analytics with Python: A Practical Real-World Approach to Gaining Actionable Insights from Your Data"](https://www.apress.com/gp/book/9781484223888), written by *Dipanjan Sarkar* and published by Apress in 2016. My role was to write so called integration code linking together different parts of the processing pipeline described in the next section. Whenever the code has been adopted, I preserved the original file and function names given by Dipanjan Sarkar. I also adopted two functions related to word cloud generation from the Jupyter notebook (https://nbviewer.jupyter.org/github/LucasTurtle/national-anthems-clustering/blob/master/Cluster_Anthems.ipynb) created by *Lucas de Sá*.

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

###Cluster 0 details:
--------------------------------------------------
Key features: ['spark', 'machine', 'regression', 'logger', 'error', 'apache', 'program', 'sparksession', 'feature', 'mllib']
Documents in this cluster:
A-Gentle-Introduction-to-Apache-Spark, Advanced_Analytics_with_Spark, Apache Spark 2.x Machine Learning Cookbook, Apache Spark 2.x for Java Developers, Apache Spark Deep Learning Cookbook

Topic #1 with weights
[('spark', 3.22), ('apache', 1.56), ('machine', 1.38), ('network', 1.37), ('system', 1.37), ('neural', 1.36), ('screenshot', 1.32), ('scala', 1.31), ('sum', 1.31), ('model', 1.31)]

###Cluster 1 details:
--------------------------------------------------
Key features: ['solr', 'query', 'facet', 'cache', 'parser', 'filter', 'content', 'index', 'folder', 'extraction']
Documents in this cluster:
Apache_Solr_Essentials, Apache_Solr_High_Performance

Topic #1 with weights
[('solr', 1.9), ('query', 1.69), ('index', 1.45), ('search', 1.26), ('apache', 1.25), ('parser', 1.24), ('performance', 1.24), ('filter', 1.22), ('result', 1.22), ('response', 1.22)]

###Cluster 2 details:
--------------------------------------------------
Key features: ['loss', 'accuracy', 'train', 'tensorflow', 'model', 'automl', 'tf', 'relu', 'encoder', 'activation']
Documents in this cluster:
Advanced Deep Learning with Keras, Advanced Deep Learning with Python, Deep Learning with TensorFlow 2 and Keras - Second Edition, Deep_Learning_for_Search, Deep_Learning_with_JavaScript, Hands-On Deep Learning for IoT

Topic #1 with weights
[('model', 2.17), ('loss', 1.95), ('iot', 1.74), ('train', 1.68), ('deep', 1.61), ('tf', 1.6), ('okun', 1.56), ('neural', 1.53), ('image', 1.51), ('input', 1.49)]

###Cluster 3 details:
--------------------------------------------------
Key features: ['elasticsearch', 'query', 'index', 'score', 'node', 'search', 'title', 'match', 'logstash', 'twitter']
Documents in this cluster:
Advanced Elasticsearch 7.0, Elasticsearch 5.x Cookbook - Third Edition, Elasticsearch 7 Quick Start Guide, Elasticsearch A Complete Guide, Elasticsearch_for_Hadoo, Learning Elastic Stack 6.0, Learning Elasticsearch, Mastering ElasticSearch

Topic #1 with weights
[('index', 2.96), ('elasticsearch', 2.48), ('query', 2.47), ('aggregation', 1.92), ('elastic', 1.65), ('es', 1.59), ('score', 1.55), ('search', 1.55), ('level', 1.54), ('hadoop', 1.53)]

A book title and its content from extracted top bigrams and trigrams are written to an SQLite database (file *documents.sqlite*). However, there is a check preventing any book to be written more than once in order to avoid duplicated records and unnecessary database growth.

## Potential future improvements
I observed that tokens from a programming code sometimes polluted clusters. This happened because many of my books contain a lot of code snippets and text pre-processing, despite being rigorous, was unable to clean up these artifacts. One potential solution of this problem could be paragraph extraction, e.g., based on some heuristics such as blank lines between paragraphs, followed by paragraph classification into code and plain text. Naturally, the latter would require a one-class or binary classifier trained on examples of code in several popular programming languages and, if a binary classifier is used, plain text. The goal is to filter out paragraphs (almost) entirely consisting of code, while leaving paragraphs with a minor fraction of code untouched as few instances of code in the whole large book would unlikely result in the high [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) score and hence, such "noisy" tokens won't do much harm to document clustering.
