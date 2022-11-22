from get_embeddings import Embeddings
from unsupervised_clustering import UnsupervisedClustering
import numpy as np
from scipy.spatial.distance import cosine
from scipy.spatial.distance import euclidean
from unsupervised_clustering import face_clustering

"""
Adds remaining photos to existing clusters
"""


# class final_groups():


def create_clustering_idx_dict(face_arrays, pred_labels):
    """
    Creates utility dictionary. Collects indices from predicted labels based on faces emb.
    If no face was detected, predicted label is considered as -1 (with other -1 outliers from dbscan).
    Return dictionary: {original id: {'true_label: 101, pred_id: index of non-zero face array, 'pred_label': clustering result}}
    """
    clustering_idx_dict = {}
    k_cluster_face = 0
    for i in range(len(face_arrays)):
        if face_arrays[i] is not None:
            clustering_idx_dict[i] = {
                # 'true_label': true_labels[i],
                'pred_id': k_cluster_face,
                'pred_label': pred_labels[k_cluster_face]
            }
            k_cluster_face += 1
        else:
            clustering_idx_dict[i] = {
                # 'true_label': true_labels[i],
                'pred_id': None,
                'pred_label': -1
            }
    return clustering_idx_dict


def create_clusters_dict(face_arrays, clustering_idx_dict):
    """Utility dictionary.
    Returns dict: {predicted cluster id: [list of original ids in that cluster]}"""

    clusters_dict = {}
    for i in range(len(face_arrays)):
        if clustering_idx_dict[i]['pred_label'] not in list(clusters_dict.keys()):
            clusters_dict[clustering_idx_dict[i]['pred_label']] = {'orig_ids': [i]}
        else:
            clusters_dict[clustering_idx_dict[i]['pred_label']]['orig_ids'].append(i)
    return clusters_dict


def add_body_centroids(clusters_dict, body_emb):
    """
    Calculating average body embedding per cluster except for outliers. Adding centroid to each cluster's dict
    """
    for cluster in clusters_dict.keys():
        if cluster != -1:
            body_emb_list = np.array(np.array(body_emb)[clusters_dict[cluster]['orig_ids']])
            cluster_centroid = np.mean(body_emb_list, axis=0)
            clusters_dict[cluster]['body_emb_centroid'] = cluster_centroid
        else:
            clusters_dict[cluster]['body_emb_centroid'] = None
    return clusters_dict


def cluster_outliers(clusters_dict, body_emb):
    """
    # For each outlier choosing closest cluster by min eucledian distance to cluster centroid
    """
    outliers_clustered = {}
    # check if not only outliers were predicted
    if (clusters_dict.keys()) != [-1]:
        for i in clusters_dict[-1]['orig_ids']:
            min_distance = np.inf

            # go thru every cluster already in place (other than -1)
            for cluster in clusters_dict.keys():
                if clusters_dict[cluster]['body_emb_centroid'] is not None:
                    # cos_d = cosine(body_emb[i], clusters_dict[cluster]['body_emb_centroid'])
                    # calculate eucl distance from current outlier to mean body embedding for that cluster
                    euc_d = euclidean(body_emb[i], clusters_dict[cluster]['body_emb_centroid'])
                    if euc_d < min_distance:
                        closest_cluster = cluster
                        min_distance = euc_d
                    outliers_clustered[i] = closest_cluster
    else:
        # if no other clusters were predicted, they are all assigned to cluster 0
        outliers_clustered = {k: 0 for k in list(clusters_dict[-1]['orig_ids'])}

    return outliers_clustered


def get_final_clusters(source_path):
    # get face and body embeddings of the source folder images
    my_emb = Embeddings(source_path)
    body_arrays, face_arrays, face_emb, body_emb = my_emb.main()

    face_emb_cleaned = np.array([emb[0] for emb in face_emb if emb is not None])  # cluster only for detected faces

    face_clusters = face_clustering(embeddings_list=face_emb_cleaned, num_components=4)
    pred_labels = face_clusters.main()

    clustering_idx_dict = create_clustering_idx_dict(face_arrays, pred_labels)
    clusters_dict = create_clusters_dict(face_arrays, clustering_idx_dict)
    clusters_dict = add_body_centroids(clusters_dict, body_emb)

    # if outliers were detected, assign them to existing clusters
    if -1 in list(clusters_dict.keys()):
        outliers_clustered = cluster_outliers(clusters_dict, body_emb)

        # add clustered outliers to clusters dict
        for item in outliers_clustered.items():
            clusters_dict[item[1]]['orig_ids'].append(item[0])

        del clusters_dict[-1]

    final_clusters = np.empty(len(body_arrays))

    # add
    for cluster in clusters_dict.keys():
        final_clusters[clusters_dict[cluster]['orig_ids']] = cluster
    final_clusters = final_clusters.astype(int)
    return my_emb.img_paths, final_clusters
