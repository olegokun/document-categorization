# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 20:42:12 2016

@author: DIP
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_similarity
import random, os
from matplotlib.font_manager import FontProperties
from collections import Counter
from wordcloud import WordCloud


# Transforms a centroids dataframe into a dictionary to be used on a WordCloud.
def centroids_dict(centroids, index):
    a = centroids.T[index].sort_values(ascending = False).reset_index().values
    centroid_dict = dict()

    for i in range(0, len(a)):
        centroid_dict.update( {a[i,0] : a[i,1]} )

    return centroid_dict


def generate_wordclouds(centroids):
    wordcloud = WordCloud(max_font_size=100, background_color = 'white')
    for i in range(0, len(centroids)):
        centroid_dict = centroids_dict(centroids, i)
        wordcloud.generate_from_frequencies(centroid_dict)
        plt.figure()
        plt.title('Cluster {}'.format(i))
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.show()


def get_cluster_data(clustering_obj, data, feature_names, num_clusters,
                     topn_features=10):

    cluster_details = {} 
    # Get cluster centroids
    ordered_centroids = clustering_obj.cluster_centers_.argsort()[:, ::-1]
    # Get key features (by TFIDF score) for each cluster
    # Get documents belonging to each cluster
    for cluster_num in range(num_clusters):
        cluster_details[cluster_num] = {}
        cluster_details[cluster_num]['cluster_num'] = cluster_num
        key_features = [feature_names[index] 
                        for index 
                        in ordered_centroids[cluster_num, :topn_features]]
        cluster_details[cluster_num]['key_features'] = key_features
        
        documents = data[data['Cluster'] == cluster_num]['Title'].values.tolist()
        cluster_details[cluster_num]['documents'] = documents
    
    return cluster_details
       
    
def print_cluster_data(cluster_data):
    # Print cluster details
    for cluster_num, cluster_details in cluster_data.items():
        print('Cluster {} details:'.format(cluster_num))
        print('-'*50)
        print('Key features:', cluster_details['key_features'])
        print('Documents in this cluster:')
        print(', '.join(cluster_details['documents']))
        print('='*70)


def plot_clusters(num_clusters, feature_matrix,
                  cluster_data, data,
                  plot_size=(16,8)):
    # Generate random color for clusters                  
    def generate_random_color():
        color = '#%06x' % random.randint(0, 0xFFFFFF)
        return color
    # Define markers for clusters    
    markers = ['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd']
    # Build cosine distance matrix
    cosine_distance = 1 - cosine_similarity(feature_matrix) 
    # Dimensionality reduction using MDS
    mds = MDS(n_components=2, dissimilarity="precomputed", 
              random_state=1)
    # Get coordinates of clusters in new low-dimensional space
    plot_positions = mds.fit_transform(cosine_distance)  
    x_pos, y_pos = plot_positions[:, 0], plot_positions[:, 1]
    # Build cluster plotting data
    cluster_color_map = {}
    cluster_name_map = {}
    for cluster_num, cluster_details in cluster_data.items():
        # Assign cluster features to unique label
        cluster_color_map[cluster_num] = generate_random_color()
        cluster_name_map[cluster_num] = ', '.join(cluster_details['key_features'][:5]).strip()
    # Map each unique cluster label with its coordinates and movies
    cluster_plot_frame = pd.DataFrame({'x': x_pos,
                                       'y': y_pos,
                                       'label': data['Cluster'].values.tolist(),
                                       'title': data['Title'].values.tolist()
                                        })
    grouped_plot_frame = cluster_plot_frame.groupby('label')
    # Set plot figure size and axes
    fig, ax = plt.subplots(figsize=plot_size) 
    ax.margins(0.05)
    # Plot each cluster using co-ordinates and movie titles
    for cluster_num, cluster_frame in grouped_plot_frame:
         marker = markers[cluster_num] if cluster_num < len(markers) \
                  else np.random.choice(markers, size=1)[0]
         ax.plot(cluster_frame['x'], cluster_frame['y'], 
                 marker=marker, linestyle='', ms=12,
                 label=cluster_name_map[cluster_num], 
                 color=cluster_color_map[cluster_num], mec='none')
         ax.set_aspect('auto')
         ax.tick_params(axis= 'x', which='both', bottom='off', top='off',        
                        labelbottom='off')
         ax.tick_params(axis= 'y', which='both', left='off', top='off',         
                        labelleft='off')
    fontP = FontProperties()
    fontP.set_size('small')    
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.01), fancybox=True, 
              shadow=True, ncol=5, numpoints=1, prop=fontP) 
    # Add labels as the film titles
    for index in range(len(cluster_plot_frame)):
        ax.text(cluster_plot_frame.iloc[index]['x'], 
                cluster_plot_frame.iloc[index]['y'], 
                cluster_plot_frame.iloc[index]['title'], size=8)  
    # Show the plot           
    plt.show() 


from sklearn.cluster import KMeans

def k_means(feature_matrix, num_clusters=3):
    km = KMeans(n_clusters=num_clusters,
                max_iter=10000)
    km.fit(feature_matrix)
    clusters = km.labels_
    return km, clusters

              
from sklearn.cluster import AffinityPropagation
              
def affinity_propagation(feature_matrix):
    
    ap = AffinityPropagation()
    ap.fit(feature_matrix.todense())
    clusters = ap.labels_          
    return ap, clusters


from scipy.cluster.hierarchy import ward, dendrogram

def ward_hierarchical_clustering(feature_matrix):
    
    cosine_distance = 1 - cosine_similarity(feature_matrix)
    linkage_matrix = ward(cosine_distance)
    return linkage_matrix

    
def plot_hierarchical_clusters(linkage_matrix, data, figure_size=(8,12)):
    # set size
    fig, ax = plt.subplots(figsize=figure_size) 
    titles = data['Title'].values.tolist()
    # plot dendrogram
    _ = dendrogram(linkage_matrix, orientation="left", labels=titles)
    plt.tick_params(axis= 'x',   
                    which='both',  
                    bottom='off',
                    top='off',
                    labelbottom='off')
    plt.tight_layout()
    plt.savefig('ward_hierachical_clusters.png', dpi=300)


def cluster_analysis(clustering_object, feature_names, titles, clusters,
                     topn_features, feature_matrix):
    
    centroids = pd.DataFrame(clustering_object.cluster_centers_)
    centroids.columns = feature_names
    generate_wordclouds(centroids)
            
    data = pd.DataFrame({'Title': titles, 'Cluster': clusters})
    # Get the total number of documents per cluster
    c = Counter(clusters)   
    print(c.items())
    if os.getenv('CLUSTERING') == "affinity":
        # Get the total number of clusters
        num_clusters = len(c)
    if os.getenv('CLUSTERING') == "kmeans":
        num_clusters = int(os.getenv('CLUSTER_NUMBER'))
    print('The number of clusters:', num_clusters)
    
    cluster_data = get_cluster_data(clustering_obj=clustering_object,
                                    data=data,
                                    feature_names=feature_names,
                                    num_clusters=num_clusters,
                                    topn_features=topn_features)         

    print_cluster_data(cluster_data) 

    plot_clusters(num_clusters=num_clusters, 
                  feature_matrix=feature_matrix,
                  cluster_data=cluster_data, 
                  data=data,
                  plot_size=(16,8)) 