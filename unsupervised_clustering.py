import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import numpy as np
from scipy.spatial.distance import cosine
import pandas as pd
import os
import pickle
import itertools
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from collections import Counter
from cluster_metrics import clustering_metrics


class UnsupervisedClustering():
    implemented_dim_reduction_methods = [PCA]
    implemented_clustering_methods = [DBSCAN]

    def __init__(self, dict_list, dim_reduction_method=PCA, clustering_method=DBSCAN, scale_again=True,
                 scale_again_type=StandardScaler()):
        """ Initializes an instance of the class """

        self.dict_list = dict_list
        self.dim_reduction_method = dim_reduction_method
        self.clustering_method = clustering_method
        self.scale_again = scale_again
        self.scale_again_type = scale_again_type
        self.dim_reduction_array = None
        self.applied_dim_reduction = False
        self.clusters = None
        self.concat_array = None

    def apply_dim_reduction(self):
        """ Applies dimensionality reduction to each dictionary in the list of dictionaries.
            Each dictionary in the list should have the following format:
            {'array':[1, 2, 3, 4, 5],
            'constant':1,
            'num_components':100,
            'is_scaled':False,
            'scale_type':StandardScaler()}
        """

        # raises error if dimensionality reduction method does not exist
        if self.dim_reduction_method not in self.implemented_dim_reduction_methods:
            raise NotImplementedError

        # if dimensionality reduction method is None return the concated array:
        if self.dim_reduction_method is None:
            for dict_ in self.dict_list:
                if self.concat_array is None:
                    self.concat_array = dict_['array']
                else:
                    self.concat_array = np.concatenate((self.concat_array, dict_['array']), axis=1)
            return self

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

        # raises error if dimensionality reduction method does not exist
        if self.dim_reduction_method not in self.implemented_dim_reduction_methods:
            raise NotImplementedError

        # gets total number of components (combined from the inserted arrays)
        num_components = sum([i['num_components'] for i in self.dict_list])

        # applies and fits dimensionality reduction method
        dim_reduction = self.dim_reduction_method(num_components)
        dim_reduction.fit(self.dim_reduction_array)

        # plots the graph with variance per component and acumulated
        plt.figure(figsize=fig_size)
        x_range = np.arange(1, num_components + 1)
        plt.plot(x_range, dim_reduction.explained_variance_ratio_)
        plt.xlabel('# principal components')
        plt.ylabel('Variance explained by component', color='C0')
        plt.twinx()

        plt.plot(x_range, dim_reduction.explained_variance_ratio_.cumsum(), color='C1')
        plt.ylabel('Cumulative variance explained', color='C1')
        plt.show()

    def apply_clustering(self, cluster_params):
        """ Applies the clustering method to the array extracted using dimensionality reduction"""

        # raises error if clustering method does not exist
        if self.clustering_method not in self.implemented_clustering_methods:
            raise NotImplementedError

        # checks whether dimensionality reduction was applied (to decide if we'll use
        # self.dim_reduction_array or just self.concat_array)

        if self.applied_dim_reduction:
            clustering_array = self.dim_reduction_array
        else:
            clustering_array = self.concat_array

        # Logic for DBSCAN clustering
        if self.clustering_method == DBSCAN:
            self.clusters = self.clustering_method(eps=cluster_params['eps'],
                                                   min_samples=cluster_params['min_samples'],
                                                   metric=cluster_params['metric'],
                                                   leaf_size=cluster_params['leaf_size']). \
                fit(clustering_array)
        return self

    def plot_scatter(self, label_type, labels=None, fig_size=(15, 10), display_outliers=True,
                     display_only_outliers=False):
        """ Plots scatter with PCA with 2 dimensions (2 first components)
        possible labels types: 'clusters' or 'true_labels'. The values should
        be provided
        """
        # raises error if dimensionality reduction method does not exist
        if self.dim_reduction_method not in self.implemented_dim_reduction_methods:
            raise NotImplementedError

        # raises error if clustering method does not exist
        if self.clustering_method not in self.implemented_clustering_methods:
            raise NotImplementedError

        # extract labels from cluster
        labels_ = sorted(Counter(labels).items())

        # gets total number of components
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

        # plot the chart with colors per cluster/ true label according to input
        plt.figure(figsize=fig_size)
        for values in labels_:
            label = values[0]
            plt.scatter(array_1[labels == label, 0],
                        array_2[labels == label, 1],
                        label=f'{label_type} {label}')

        plt.xlabel('Component 0')
        plt.ylabel('Component 1')
        plt.legend()
        plt.title(f'Points clustered with DBScan and PCA (k: {num_components}), colored by {label_type}')
        plt.show()

    def plot_outliers_with_true_label(self, true_labels_, fig_size=(15, 10), outlier_label=-1):
        """ Plots scatter with PCA with 2 dimensions (2 first components)
        containing the outliers that the cluster identified, but coloured with their true labels
        """

        # raises error if dimensionality reduction method does not exist
        if self.dim_reduction_method not in self.implemented_dim_reduction_methods:
            raise NotImplementedError

        # raises error if clustering method does not exist
        if self.clustering_method not in self.implemented_clustering_methods:
            raise NotImplementedError

        # extract labels from cluster
        labels = self.clusters.labels_
        labels_ = sorted(Counter(labels).items())

        # gets true label where the clustering labels are outliers
        true_labels_ = np.array(true_labels_)[[labels == outlier_label]]

        # gets total number of components
        num_components = sum([i['num_components'] for i in self.dict_list])

        # if there is only one array, uses first 2 components from the array
        # else it takes the first components of the array in the first dict
        # and the first component of the array in the second dict
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


class UnsupervisedGridSearch:
    def __init__(self, dict_list, clustering_params, true_labels, chosen_metric='AMI'):
        """ Initializes an instance of the class """
        self.dict_list = dict_list
        self.clustering_params = clustering_params
        self.true_labels = true_labels
        self.chosen_metric = chosen_metric
        self._best_dim_red_combination = None
        self._best_cluster_combination = None
        self.best_combination = None
        self._best_metric = None
        self.grid_search_results = []
        self.clustering_combinations = []
        self.dim_reduction_combinations = []

    def _create_clustering_combinations(self):
        """ Given a dictionary in the format clustering_params = {'eps_values_list': [...], 'min_samples_values_list': [...], 'metrics_values_list': [...], 'leaf_size_values_list':  [...]}
        creates a list with dictionaries with all the possible combinations of parameters """

        clustering_hyperparams_list = [i for i in self.clustering_params.values()]
        clustering_combinations_list = list(itertools.product(*clustering_hyperparams_list))

        for clustering_combination in clustering_combinations_list:
            eps, min_samples, metric, leaf_size = clustering_combination
            dict_ = {'eps': eps, 'min_samples': min_samples, 'metric': metric, 'leaf_size': leaf_size}
            self.clustering_combinations.append(dict_)

        return self

    def _create_dim_reduction_combinations(self):
        """ Given a list of dictionary with the following format  [{'array': [embedding_array],'constant':[...],'num_components':[...], 'is_scaled':[...], 'scale_type':[]}]
        Creates a list of dictionaries with all the possible combinations. The list can contain multiple dictionaries.
        The value for the key 'array' should be a numpy array, it is not a parameter for the combinations"""

        combinations_list = []

        for idx, value_ in enumerate(self.dict_list):
            array_ = self.dict_list[idx]['array']
            constant_list = self.dict_list[idx]['constant']
            num_components_list = self.dict_list[idx]['num_components']
            is_scaled_list = self.dict_list[idx]['is_scaled']
            scale_type_list = self.dict_list[idx]['scale_type']

            dim_reduction_hyperparams = [array_, constant_list, num_components_list, is_scaled_list, scale_type_list]
            # creates combination
            combinations_list.append(list(itertools.product(*dim_reduction_hyperparams)))

        # transforms in list of list of dicts
        # for now we have the combinations for each of the arrays, we need the actual combination of both
        # is a list of tuples
        dim_reduction_combinations_list = list(itertools.product(*combinations_list))

        # transform the list of tuples into list of list of dicts
        self.dim_reduction_combinations = []
        for list_ in dim_reduction_combinations_list:
            combination = []
            for tup in list_:
                array, constant, num_component, is_scaled, scale_type = tup
                combination.append({'array': array,
                                    'constant': constant,
                                    'num_components': num_component,
                                    'is_scaled': is_scaled,
                                    'scale_type': scale_type})
            self.dim_reduction_combinations.append(combination)

        return self

    def apply_grid_search(self):
        """ Applies grid search with all the parameters obtained from the combinations"""
        # create combinations
        if len(self.clustering_combinations) == 0:
            self._create_clustering_combinations()

        if len(self.dim_reduction_combinations) == 0:
            self._create_dim_reduction_combinations()

        # list of index combinations to be able to traceback to best combination of results
        idx_list = []
        # iterate over the dim reduction combinations
        for dim_reduction_idx, dim_reduction_params in enumerate(self.dim_reduction_combinations):
            self.uc = UnsupervisedClustering(dict_list=dim_reduction_params)
            self.uc.apply_dim_reduction()

            # build results_dict
            for cluster_idx, cluster_params in enumerate(self.clustering_combinations):
                # apply clustering with the parameters and add to results_dict
                self.uc.apply_clustering(cluster_params)
                results_dict = clustering_metrics(self.true_labels, self.uc.clusters.labels_)

                # add cluster info
                for key_ in cluster_params.keys():
                    results_dict[key_] = cluster_params[key_]

                # add also information about the embeddings/ pca (constant, num_components, is_scaled_scale_type)
                for idx, dict_ in enumerate(dim_reduction_params, 1):
                    results_dict[f'constant_{idx}'] = dict_['constant']
                    results_dict[f'num_components_{idx}'] = dict_['num_components']
                    results_dict[f'is_scaled_{idx}'] = dict_['is_scaled']
                    results_dict[f'scale_type_{idx}'] = dict_['scale_type']

                # adds the dict to the list of results
                self.grid_search_results.append(results_dict)
                idx_list.append((dim_reduction_idx, cluster_idx))

        # gets best metric in user friendly format
        idx_best_metric = \
        pd.DataFrame(self.grid_search_results).sort_values(by=[self.chosen_metric], ascending=False).index[0]
        self.best_combination = self.grid_search_results[idx_best_metric]

        # traceback to best dim_reduction and cluster params
        best_dim_reduction_idx, best_cluster_idx = idx_list[idx_best_metric]
        self._best_dim_red_combination = self.dim_reduction_combinations[best_dim_reduction_idx]
        self._best_cluster_combination = self.clustering_combinations[best_cluster_idx]

        # runs best combination
        self._run_best_metric()

        return self

    def _run_best_metric(self):
        """ runs for the combination that returns the best metric"""
        self.uc = UnsupervisedClustering(dict_list=self._best_dim_red_combination)
        self.uc.apply_dim_reduction()
        self.uc.apply_clustering(self._best_cluster_combination)
        self._best_metric = clustering_metrics(self.true_labels, self.uc.clusters.labels_)
        return self