import numpy as np
from sklearn import metrics
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score, rand_score


def clustering_metrics(true_labels, pred_labels):
    """
    Takes a list of true (actual) labels and predicted labels.
     Returns dictionary of metrics: purity, NMI, AMI, number of predicted clusters, whether outliers were detected.
     AMI is a key metric.
     This function MUST BE USED in all modeling tests
    """

    # Purity
    contingency_matrix = metrics.cluster.contingency_matrix(true_labels, pred_labels)
    purity = np.sum(np.amax(contingency_matrix, axis=1)) / np.sum(contingency_matrix)

    # Normalized Mutual Information (NMI)
    NMI = normalized_mutual_info_score(true_labels, pred_labels)

    # Adjusted Mutual Information (AMI)
    AMI = adjusted_mutual_info_score(true_labels, pred_labels)

    # Rand Index
    RI = rand_score(true_labels, pred_labels)

    # number of predicted clusters:
    num_pred_clusters = np.count_nonzero(np.unique(pred_labels) != -1)

    # for dbscan only - whether outliers were predicted or not
    outl_detected = -1 in pred_labels

    result_dict = {'purity': np.around(purity, 3),
                   'NMI': np.around(NMI, 3),
                   'AMI': np.around(AMI, 3),
                   'RI': np.around(RI, 3),
                   'num_pred_clusters': num_pred_clusters,
                   'outl_detected': outl_detected
                   }

    return result_dict

