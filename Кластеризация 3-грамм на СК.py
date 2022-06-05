import numpy as np
import pandas as pd
from sklearn.metrics import calinski_harabasz_score
import pickle
from sklearn.cluster import KMeans
from scipy.special import gamma
from sklearn.neighbors import KDTree
from collections import defaultdict
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform, euclidean
from itertools import combinations, product


class Wishart:
    def __init__(self, wishart_neighbors, significance_level):
        self.wishart_neighbors = wishart_neighbors  # Number of neighbors
        self.significance_level = significance_level  # Significance level

    def fit(self, X):
        from sklearn.neighbors import KDTree
        kdt = KDTree(X, metric='euclidean')

        #add one because you are your neighb.
        distances, neighbors = kdt.query(X, k = self.wishart_neighbors + 1, return_distance = True)
        neighbors = neighbors[:, 1:]


        distances = distances[:, -1]
        indexes = np.argsort(distances)
        
        size, dim = X.shape

        self.object_labels = np.zeros(size, dtype = int) - 1
        
        # ADDED FOR SCORES
        dist = squareform(pdist(X))    # matrix of distances
        self.dist_ = np.array(dist)
        self.dk_ = distances

        #index in tuple
        #min_dist, max_dist, flag_to_significant
        self.clusters = np.array([(1., 1., 0)])
        self.clusters_to_objects = defaultdict(list)
        print('Start clustering')

        for index in indexes:
            neighbors_clusters =\
                np.concatenate([self.object_labels[neighbors[index]], self.object_labels[neighbors[index]]])
            unique_clusters = np.unique(neighbors_clusters).astype(int)
            unique_clusters = unique_clusters[unique_clusters != -1]


            if len(unique_clusters) == 0:
                self._create_new_cluster(index, distances[index])
            else:
                max_cluster = unique_clusters[-1]
                min_cluster = unique_clusters[0]
                if max_cluster == min_cluster:
                    if self.clusters[max_cluster][-1] < 0.5:
                        self._add_elem_to_exist_cluster(index, distances[index], max_cluster)
                    else:
                        self._add_elem_to_noise(index)
                else:
                    my_clusters = self.clusters[unique_clusters]
                    flags = my_clusters[:, -1]
                    if np.min(flags) > 0.5:
                        self._add_elem_to_noise(index)
                    else:
                        significan = np.power(my_clusters[:, 0], -dim) - np.power(my_clusters[:, 1], -dim)
                        significan *= self.wishart_neighbors
                        significan /= size
                        significan /= np.power(np.pi, dim / 2)
                        significan *= gamma(dim / 2 + 1)
                        significan_index = significan >= self.significance_level

                        significan_clusters = unique_clusters[significan_index]
                        not_significan_clusters = unique_clusters[~significan_index]
                        significan_clusters_count = len(significan_clusters)
                        if significan_clusters_count > 1 or min_cluster == 0:
                            self._add_elem_to_noise(index)
                            self.clusters[significan_clusters, -1] = 1
                            for not_sig_cluster in not_significan_clusters:
                                if not_sig_cluster == 0:
                                    continue

                                for bad_index in self.clusters_to_objects[not_sig_cluster]:
                                    self._add_elem_to_noise(bad_index)
                                self.clusters_to_objects[not_sig_cluster].clear()
                        else:
                            for cur_cluster in unique_clusters:
                                if cur_cluster == min_cluster:
                                    continue

                                for bad_index in self.clusters_to_objects[cur_cluster]:
                                    self._add_elem_to_exist_cluster(bad_index, distances[bad_index], min_cluster)
                                self.clusters_to_objects[cur_cluster].clear()

                            self._add_elem_to_exist_cluster(index, distances[index], min_cluster)

        return self.clean_data()

    def clean_data(self):
        unique = np.unique(self.object_labels)
        index = np.argsort(unique)
        if unique[0] != 0:
            index += 1
        true_cluster = {unq :  index for unq, index in zip(unique, index)}
        result = np.zeros(len(self.object_labels), dtype = int)
        for index, unq in enumerate(self.object_labels):
            result[index] = true_cluster[unq]
        return result

    def _add_elem_to_noise(self, index):
        self.object_labels[index] = 0
        self.clusters_to_objects[0].append(index)

    def _create_new_cluster(self, index, dist):
        self.object_labels[index] = len(self.clusters)
        self.clusters_to_objects[len(self.clusters)].append(index)
        self.clusters = np.append(self.clusters, [(dist, dist, 0)], axis = 0)

    def _add_elem_to_exist_cluster(self, index, dist, cluster_label):
        self.object_labels[index] = cluster_label
        self.clusters_to_objects[cluster_label].append(index)
        self.clusters[cluster_label][0] = min(self.clusters[cluster_label][0], dist)
        self.clusters[cluster_label][1] = max(self.clusters[cluster_label][1], dist)


class PreTrainWishart:
    def __init__(self, wishart_neighbors, significance_level, distances, neighbors):
        self.wishart_neighbors = wishart_neighbors  # Number of neighbors
        self.significance_level = significance_level  # Significance level
        self.distances = distances
        self.neighbors = neighbors

    def fit(self, X):
        from sklearn.neighbors import KDTree
        kdt = KDTree(X, metric='euclidean')

        #add one because you are your neighb.
        neighbors = self.neighbors[:, 1 : self.wishart_neighbors + 1]
        distances = self.distances[:, self.wishart_neighbors]
        indexes = np.argsort(distances)
        
        size, dim = X.shape

        self.object_labels = np.zeros(size, dtype = int) - 1

        #index in tuple
        #min_dist, max_dist, flag_to_significant
        self.clusters = np.array([(1., 1., 0)])
        self.clusters_to_objects = defaultdict(list)

        for index in indexes:
            neighbors_clusters =\
                np.concatenate([self.object_labels[neighbors[index]], self.object_labels[neighbors[index]]])
            unique_clusters = np.unique(neighbors_clusters).astype(int)
            unique_clusters = unique_clusters[unique_clusters != -1]


            if len(unique_clusters) == 0:
                self._create_new_cluster(index, distances[index])
            else:
                max_cluster = unique_clusters[-1]
                min_cluster = unique_clusters[0]
                if max_cluster == min_cluster:
                    if self.clusters[max_cluster][-1] < 0.5:
                        self._add_elem_to_exist_cluster(index, distances[index], max_cluster)
                    else:
                        self._add_elem_to_noise(index)
                else:
                    my_clusters = self.clusters[unique_clusters]
                    flags = my_clusters[:, -1]
                    if np.min(flags) > 0.5:
                        self._add_elem_to_noise(index)
                    else:
                        significan = np.power(my_clusters[:, 0], -dim) - np.power(my_clusters[:, 1], -dim)
                        significan *= self.wishart_neighbors
                        significan /= size
                        significan /= np.power(np.pi, dim / 2)
                        significan *= gamma(dim / 2 + 1)
                        significan_index = significan >= self.significance_level

                        significan_clusters = unique_clusters[significan_index]
                        not_significan_clusters = unique_clusters[~significan_index]
                        significan_clusters_count = len(significan_clusters)
                        if significan_clusters_count > 1 or min_cluster == 0:
                            self._add_elem_to_noise(index)
                            self.clusters[significan_clusters, -1] = 1
                            for not_sig_cluster in not_significan_clusters:
                                if not_sig_cluster == 0:
                                    continue

                                for bad_index in self.clusters_to_objects[not_sig_cluster]:
                                    self._add_elem_to_noise(bad_index)
                                self.clusters_to_objects[not_sig_cluster].clear()
                        else:
                            for cur_cluster in unique_clusters:
                                if cur_cluster == min_cluster:
                                    continue

                                for bad_index in self.clusters_to_objects[cur_cluster]:
                                    self._add_elem_to_exist_cluster(bad_index, distances[bad_index], min_cluster)
                                self.clusters_to_objects[cur_cluster].clear()

                            self._add_elem_to_exist_cluster(index, distances[index], min_cluster)

        return self.clean_data()

    def clean_data(self):
        unique = np.unique(self.object_labels)
        index = np.argsort(unique)
        if unique[0] != 0:
            index += 1
        true_cluster = {unq :  index for unq, index in zip(unique, index)}
        result = np.zeros(len(self.object_labels), dtype = int)
        for index, unq in enumerate(self.object_labels):
            result[index] = true_cluster[unq]
        return result

    def _add_elem_to_noise(self, index):
        self.object_labels[index] = 0
        self.clusters_to_objects[0].append(index)

    def _create_new_cluster(self, index, dist):
        self.object_labels[index] = len(self.clusters)
        self.clusters_to_objects[len(self.clusters)].append(index)
        self.clusters = np.append(self.clusters, [(dist, dist, 0)], axis = 0)

    def _add_elem_to_exist_cluster(self, index, dist, cluster_label):
        self.object_labels[index] = cluster_label
        self.clusters_to_objects[cluster_label].append(index)
        self.clusters[cluster_label][0] = min(self.clusters[cluster_label][0], dist)
        self.clusters[cluster_label][1] = max(self.clusters[cluster_label][1], dist)

def divide(data, labels):
    clusters = set(labels)
    clusters_data = []
    for cluster in clusters:
        clusters_data.append(data[labels == cluster, :])
    return clusters_data

def get_centroids(clusters):
    centroids = []
    for cluster_data in clusters:
        centroids.append(cluster_data.mean(axis=0))
    return centroids

def SST(data):
    c = get_centroids([data])
    return ((data - c) ** 2).sum()

def SSE(clusters, centroids):
    result = 0
    for cluster, centroid in zip(clusters, centroids):
        result += ((cluster - centroid) ** 2).sum()
    return result

# Clear the store before running each time
within_cluster_dist_sum_store = {}
def within_cluster_dist_sum(cluster, centroid, cluster_id):
    if cluster_id in within_cluster_dist_sum_store:
        return within_cluster_dist_sum_store[cluster_id]
    else:
        result = (((cluster - centroid) ** 2).sum(axis=1)**.5).sum()
        within_cluster_dist_sum_store[cluster_id] = result
    return result

def RMSSTD(data, clusters, centroids):
    df = data.shape[0] - len(clusters)
    attribute_num = data.shape[1]
    return (SSE(clusters, centroids) / (attribute_num * df)) ** .5

# equal to separation / (cohesion + separation)
def RS(data, clusters, centroids):
    sst = SST(data)
    sse = SSE(clusters, centroids)
    return (sst - sse) / sst

def DB_find_max_j(clusters, centroids, i):
    max_val = 0
    max_j = 0
    for j in range(len(clusters)):
        if j == i:
            continue
        cluster_i_stat = within_cluster_dist_sum(clusters[i], centroids[i], i) / clusters[i].shape[0]
        cluster_j_stat = within_cluster_dist_sum(clusters[j], centroids[j], j) / clusters[j].shape[0]
        val = (cluster_i_stat + cluster_j_stat) / (((centroids[i] - centroids[j]) ** 2).sum() ** .5)
        if val > max_val:
            max_val = val
            max_j = j
    return max_val

def DB(data, clusters, centroids):
    result = 0
    for i in range(len(clusters)):
        result += DB_find_max_j(clusters, centroids, i)
    return result / len(clusters)

def XB(data, clusters, centroids):
    sse = SSE(clusters, centroids)
    min_dist = ((centroids[0] - centroids[1]) ** 2).sum()
    for centroid_i, centroid_j in list(product(centroids, centroids)):
        if (centroid_i - centroid_j).sum() == 0:
            continue
        dist = ((centroid_i - centroid_j) ** 2).sum()
        if dist < min_dist:
            min_dist = dist
    return sse / (data.shape[0] * min_dist)

def inter_cluster_distances(labels, distances, method='nearest'):
    """Calculates the distances between the two nearest points of each cluster.
    :param labels: a list containing cluster labels for each of the n elements
    :param distances: an n x n numpy.array containing the pairwise distances between elements
    :param method: `nearest` for the distances between the two nearest points in each cluster, or `farthest`
    """
    
    
    CLUSTER_DISTANCE_METHODS = ['nearest', 'farthest']
    
    if method not in CLUSTER_DISTANCE_METHODS:
        raise ValueError(
            'method must be one of {}'.format(CLUSTER_DISTANCE_METHODS))

    if method == 'nearest':
        return __cluster_distances_by_points(labels, distances)
    elif method == 'farthest':
        return __cluster_distances_by_points(labels, distances, farthest=True)


def __cluster_distances_by_points(labels, distances, farthest=False):
    n_unique_labels = len(np.unique(labels))
    cluster_distances = np.full((n_unique_labels, n_unique_labels),
                                float('inf') if not farthest else 0)

    np.fill_diagonal(cluster_distances, 0)

    for i in np.arange(0, len(labels) - 1):
        for ii in np.arange(i, len(labels)):
            if labels[i] != labels[ii] and (
                (not farthest and
                 distances[i, ii] < cluster_distances[labels[i], labels[ii]])
                    or
                (farthest and
                 distances[i, ii] > cluster_distances[labels[i], labels[ii]])):
                cluster_distances[labels[i], labels[ii]] = cluster_distances[
                    labels[ii], labels[i]] = distances[i, ii]
    return cluster_distances


def diameter(labels, distances, method='farthest'):
    """Calculates cluster diameters
    :param labels: a list containing cluster labels for each of the n elements
    :param distances: an n x n numpy.array containing the pairwise distances between elements
    :param method: either `mean_cluster` for the mean distance between all elements in each cluster, or `farthest` for the distance between the two points furthest from each other
    """
    DIAMETER_METHODS = ['mean_cluster', 'farthest']
    if method not in DIAMETER_METHODS:
        raise ValueError('method must be one of {}'.format(DIAMETER_METHODS))

    n_clusters = len(np.unique(labels))
    diameters = np.zeros(n_clusters)

    if method == 'mean_cluster':
        for i in range(0, len(labels) - 1):
            for ii in range(i + 1, len(labels)):
                if labels[i] == labels[ii]:
                    diameters[labels[i]] += distances[i, ii]

        for i in range(len(diameters)):
            diameters[i] /= sum(labels == i)

    elif method == 'farthest':
        for i in range(0, len(labels) - 1):
            for ii in range(i + 1, len(labels)):
                if labels[i] == labels[ii] and distances[i, ii] > diameters[
                        labels[i]]:
                    diameters[labels[i]] = distances[i, ii]
    return diameters


def dunn(data, labels, diameter_method='farthest',
         cdist_method='nearest'):
    distances = euclidean_distances(data)
    labels = LabelEncoder().fit(labels).transform(labels)

    ic_distances = inter_cluster_distances(labels, distances, cdist_method)
    min_distance = min(ic_distances[ic_distances.nonzero()])
    max_diameter = max(diameter(labels, distances, diameter_method))

    return min_distance / max_diameter

def mod_hubert_statistic(data, clusters, centroids):
    index = 0 
    for i in range(len(clusters)):
        for j in range(len(clusters)):
            if i == j:
                continue
            
            d_c = np.linalg.norm(np.array(centroids[i]) - np.array(centroids[j]))
            for x in clusters[i]:
                for y in clusters[j]:
                    index += np.linalg.norm(np.array(x) - np.array(y)) * d_c
    return index * 2 /(len(data) * (len(data) - 1))

def i_index(data, clusters, centroids):
    c = data.mean(axis=0)
    sum_d_c = np.sum(np.linalg.norm(data - c, axis=1))
    m = 0
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            m = max(m, np.linalg.norm(centroids[i] - centroids[j]))
    s = 0
    for i in range(len(clusters)):
        s += np.sum(np.linalg.norm(clusters[i] - centroids[i], axis=1))
    return (sum_d_c * m / (len(clusters) * s))**len(data[0])

def Sep(labels, k, dk, dist):
    clusters = sorted(set(labels))
    max_sep = None
    for cluster in clusters:
        cluster_data = dist[labels == cluster]
        cluster_data = cluster_data[:, labels != cluster]
        cluster_dk = dk[labels == cluster]
        sep = len(cluster_data[cluster_data <= np.c_[cluster_dk]]) / (k * cluster_data.shape[0])
        if max_sep is None or max_sep < sep:
            max_sep = sep
    return max_sep

def Com(labels, dist):
    clusters = sorted(set(labels))
    com = 0
    max_com = 0
    for cluster in clusters:
        cluster_data = dist[labels == cluster]
        cluster_data = cluster_data[:, labels == cluster]
        n_i = cluster_data.shape[0]
#        print(n_i, cluster_data.sum())
        if n_i > 1:
            cur_sum = 2 * cluster_data.sum() / (n_i * (n_i - 1))
            com += cur_sum
            if max_com < cur_sum:
                max_com = cur_sum
    return com, max_com

def CVNN(labels, k, dk, dist):
    com, max_com = Com(labels, dist)
    return Sep(labels, k, dk, dist) + com / max_com 

def get_validation_scores(data, labels, k, dk, dist):
    within_cluster_dist_sum_store.clear()
    
    clusters = divide(data, labels)
    centroids = get_centroids(clusters)
    
    scores = {}
#     scores['1 RMSSTD'] = RMSSTD(data, clusters, centroids)
    scores['RS'] = RS(data, clusters, centroids)
#     scores['3 H'] = mod_hubert_statistic(data, clusters, centroids)
    scores['CH'] = calinski_harabasz_score(data, labels)
#     scores['5 I'] = i_index(data, clusters, centroids)
#     scores['6 D'] = dunn(data, labels)
#     scores['7 S'] = silhouette_score(data, labels)
#     scores['8 DB'] = DB(data, clusters, centroids)
    scores['XB'] = XB(data, clusters, centroids)
#     scores['10 SD'] = SD(data, labels)
#     scores['11 S_Dbw'] = S_Dbw(data, labels)
#     scores['12 CVNN'] = CVNN(labels, k, dk, dist)
    return scores

# Скачиваем словарь слов - эмбедингов
dictionary_3gr = pickle.load(open("dict_emb_gr_cpu.pkl", 'rb'))

# Достаем эмбединги
dict_3gr = dictionary_3gr.values()
dict_3gr = np.array(list(dict_3gr))
small_dict1 = np.array(pd.DataFrame(dict_3gr).loc[:10000])

words_3gr = dictionary_3gr.keys()
words_3gr = np.array(list(words_3gr))
small_words1 = np.array(pd.DataFrame(words_3gr).loc[:10000])

# Запустим алгоритм кластеризации Wishart на этой подвыборке
datasets_clustering = []
datasets_names = []
num_clusters = 5
k = 2
while num_clusters >= 5:
    WC = Wishart(k, 0.001)
    res = WC.fit(small_dict1)
    num_clusters = np.max(res)
    print(num_clusters)
    datasets_clustering.append((small_dict1, WC))
    datasets_names += ['k = ' + str(k)]
    k += 5

# WC = Wishart(5, 0.001)
# res = WC.fit(small_dict1)
# num_clusters = np.max(res)
# datasets_clustering.append((small_dict1, WC))
# datasets_names += ['k = ' + str(5)]
    

scores_list = []
dict_list = [] # список словарей 
for (data, WC), data_name in zip(datasets_clustering, datasets_names):
    if(len(set(WC.object_labels)) == 1):
        print('Set with one cluster')
        continue
    scores = get_validation_scores(data, WC.object_labels, k=10, dk=WC.dk_, dist=WC.dist_)
    scores_list += [scores]
    dict_ = {}
    for i in set(WC.object_labels):
        dict_[i] =  small_words1[WC.object_labels==i]
    dict_list += [dict_]
    
a_file = open("3gr_wordlist_BERT.pkl", "wb")
pickle.dump(dict_list, a_file)
a_file.close()

b_file = open("3gr_metrics_BERT.pkl", "wb")
pickle.dump(scores_list, b_file)
b_file.close()