import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import numpy as np
from numpy import asarray
from scipy.spatial.distance import cosine
import pandas as pd
import os
import pickle
import itertools
from sklearn.cluster import DBSCAN
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from collections import Counter


class UnsupervisedClustering():
    implemented_dim_reduction_methods = [PCA]
    implemented_clustering_methods = [DBSCAN]

    def __init__(self, dict_list, clustering_params, dim_reduction_method=PCA, clustering_method=DBSCAN,
                 scale_again=True, scale_again_type=StandardScaler()):
        """ Initializes an instance of the class """

        self.dict_list = dict_list
        self.clustering_params = clustering_params
        self.dim_reduction_method = dim_reduction_method
        self.clustering_method = clustering_method
        self.scale_again = scale_again
        self.scale_again_type = scale_again_type
        self.dim_reduction_array = None
        self.applied_dim_reduction = False
        self.clusters = None

    def apply_dim_reduction(self):
        """ Applies dimensionality reduction to each dictionary in the list of dictionaries.
            Each dictionary in the list should have the following format:
            {'array':[1, 2, 3, 4, 5],
            'constant':1,
            'num_components':100,
            'is_scaled':False,
            'scale_type':StandardScaler()}
        """

        if self.dim_reduction_method not in self.implemented_dim_reduction_methods:
            raise NotImplementedError

        # iterates over list of dictionaries, and apply scaling and PCA according to what is specified in each dictionary
        for dict_ in self.dict_list:
            # multiply array by their constants
            dict_['array'] = dict_['array'] * dict_['constant']

            # creates scaled array
            if dict_['is_scaled']:
                dict_['scaled_array'] = dict_['array']
            else:
                dict_['scaled_array'] = dict_['scale_type'].fit_transform(dict_['array'])

            # runs pca and add the created array to the dictionary
            pca = self.dim_reduction_method(n_components=dict_['num_components'])
            dict_['pca_array'] = pca.fit_transform(dict_['scaled_array'])

            # creates the main PCA array which is the concatenation of all the previous arrays
            # add to pca_array
            if self.dim_reduction_array is None:
                self.dim_reduction_array = dict_['pca_array']
            else:
                self.dim_reduction_array = np.concatenate((self.dim_reduction_array, dict_['pca_array']), axis=1)

        # if specified, scales again
        if self.scale_again:
            self.dim_reduction_array = self.scale_again_type.fit_transform(self.dim_reduction_array)

        # turns to the indication if PCA was finished
        self.applied_dim_reduction = True

        return self

    def explain_dim_reduction(self, fig_size=(15, 10)):
        """ Plots the explanation of the dimensionality reduction method """

        if self.dim_reduction_method not in self.implemented_dim_reduction_methods:
            raise NotImplementedError

        num_components = sum([i['num_components'] for i in self.dict_list])

        pca = self.dim_reduction_method(num_components)
        pca.fit(self.dim_reduction_array)

        plt.figure(figsize=fig_size)
        x_range = np.arange(1, num_components + 1)
        plt.plot(x_range, pca.explained_variance_ratio_)
        plt.xlabel('# principal components')
        plt.ylabel('Variance explained by component', color='C0')
        plt.twinx()

        plt.plot(x_range, pca.explained_variance_ratio_.cumsum(), color='C1')
        plt.ylabel('Cumulative variance explained', color='C1')
        plt.show()

    def apply_clustering(self, cluster_params):
        if self.clustering_method not in self.implemented_clustering_methods:
            raise NotImplementedError

        self.clusters = DBSCAN(eps=cluster_params['eps'],
                               min_samples=cluster_params['min_samples'],
                               metric=cluster_params['metric'],
                               leaf_size=cluster_params['leaf_size']). \
            fit(self.dim_reduction_array)
        return self

    def plot_scatter(self, label_type, labels=None, fig_size=(15, 10), display_outliers=True,
                     display_only_outliers=False):
        """ Plots scatter with PCA with 2 dimensions (2 first components)
        possible labels types: 'clusters' or 'true_labels'. The values should
        be provided
        """

        if self.dim_reduction_method not in self.implemented_dim_reduction_methods:
            raise NotImplementedError

        if self.clustering_method not in self.implemented_clustering_methods:
            raise NotImplementedError

        labels_ = sorted(Counter(self.clusters.labels_).items())
        num_components = sum([i['num_components'] for i in self.dict_list])

        # removes outlier (label=-1) from plot
        if not display_outliers:
            labels_ = [i for i in labels_ if i[0] > -1]
        # keeps only outliers
        if display_only_outliers:
            labels_ = [i for i in labels_ if i[0] == -1]

        # assings values to arrays
        if len(self.dict_list) == 1:
            array_1 = self.dict_list[0]['pca_array']
            array_2 = self.dict_list[0]['pca_array']
        else:
            array_1 = self.dict_list[0]['pca_array']
            array_2 = self.dict_list[1]['pca_array']

        plt.figure(figsize=fig_size)
        for values in labels_:
            label = values[0]
            plt.scatter(array_1[labels == label, 0],
                        array_2[labels == label, 1],
                        label=f'Cluster {label}')

        plt.xlabel('Component 0')
        plt.ylabel('Component 1')
        plt.legend()
        plt.title(f'Points clustered with DBScan and PCA (k: {num_components}), colored by {label_type}')
        plt.show()

    def plot_outliers_with_true_label(self, true_labels_, fig_size=(15, 10), outlier_label=-1):
        """ Plots scatter with PCA with 2 dimensions (2 first components)
        containing the outliers that the cluster identified, but coloured with their true labels
        """

        if self.dim_reduction_method not in self.implemented_dim_reduction_methods:
            raise NotImplementedError

        if self.clustering_method not in self.implemented_clustering_methods:
            raise NotImplementedError

        labels = self.clusters.labels_
        labels_ = sorted(Counter(labels).items())
        true_labels_ = np.array(true_labels_)[[labels == outlier_label]]
        num_components = sum([i['num_components'] for i in self.dict_list])

        if len(self.dict_list) == 1:
            array_1 = self.dict_list[0]['pca_array'][labels == outlier_label, 0]
            array_2 = self.dict_list[0]['pca_array'][labels == outlier_label, 1]
        else:
            array_1 = self.dict_list[0]['pca_array'][labels == outlier_label, 0]
            array_2 = self.dict_list[1]['pca_array'][labels == outlier_label, 0]

        plt.figure(figsize=fig_size)
        for true_label_ in set(true_labels_):
            plt.scatter(array_1[true_labels_ == true_label_],
                        array_2[true_labels_ == true_label_],
                        label=f'Cluster {true_label_}')

        plt.xlabel('Component 0')
        plt.ylabel('Component 1')
        plt.legend()
        plt.title(f'Outlier points clustered with DBScan and PCA (k: {num_components}), colored by true label')
        plt.show()

