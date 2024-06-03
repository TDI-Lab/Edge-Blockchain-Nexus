import pandas as pd
import csv
from sklearn.metrics import *
from scipy.spatial.distance import *


def get_clustering_labels(Y, num, method_id):
    if method_id == "Hierarchical":
        # Perform hierarchical clustering
        linkage_matrix = linkage(Y, method='single')
        cluster_labels = fcluster(linkage_matrix, num, criterion='maxclust')
    elif method_id == "Hierarchical_ward":
        # Perform ward hierarchical clustering
        linkage_ward_matrix = linkage(Y, method='ward')
        cluster_labels = fcluster(linkage_ward_matrix, num, criterion='maxclust')
    elif method_id == "Kmeans":
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=num, random_state=0, n_init=10)
        cluster_labels = kmeans.fit_predict(Y)
        cluster_centers = kmeans.cluster_centers_
    elif method_id == "Birch":
        # Perform Birch clustering
        birch_cluster = Birch(n_clusters=num, threshold=0.1)
        birch_cluster.fit_predict(Y)
        cluster_labels = birch_cluster.labels_
    elif method_id == "Agglomerative":
        # Perform Agglomerative clustering
        agglomerative_cluster = AgglomerativeClustering(n_clusters=num)
        agglomerative_cluster.fit_predict(Y)
        cluster_labels = agglomerative_cluster.labels_
    else:
        print("No Method!")
        return []
    return cluster_labels


def main():
    columns_to_reduce = given_columns
    new_dataframe_path = 'output/Clustering_feature_reduction/binary_matrix_' + str(threshold) + '.csv'

    # Read the binary matrix for all dimensions
    data_path = 'output/binary_matrix.csv'
    data = pd.read_csv(data_path)
    # Reduce the features
    new_dataframe = pd.DataFrame()
    for col in columns_brief:
        attr_in_dim = [attr for attr in data.columns if col in attr]
        dataframe_dim = data[attr_in_dim]
        data_sum_series = dataframe_dim.sum()
        if col in columns_to_reduce:
            threshold = int((1 - threshold) * len(data_sum_series))
        else:
            threshold = 0
        selected_features = data_sum_series.sort_values(ascending=True)[threshold:]
        dataframe_dim = dataframe_dim[selected_features.index]
        new_dataframe = pd.concat([new_dataframe, dataframe_dim], axis=1)
    # write the file
    new_dataframe.to_csv(new_dataframe_path, index=False)
    print(f"The number of features after reduction is: {new_dataframe.shape[1]}")
    print(f"The number of paper after reduction is: {new_dataframe.shape[0]}")

    # Get the dataframe
    new_dataframe = pd.read_csv(new_dataframe_path)
    data_to_store = []

    # Acquire the dimension
    dimension_file_path = 'output/Reduced_dimension.csv'
    dimension_file = pd.read_csv(dimension_file_path)
    for i in range(dimension_file.shape[1]):
        given_columns_to_test = dimension_file[str(i + 1)].dropna().values

        # Find the features of the given dimensions for the dataframe
        attributes = []
        for column in given_columns_to_test:
            attributes = attributes + [attr for attr in new_dataframe.columns if column in attr]
        data_frame = new_dataframe[attributes]

        # Clustering and Calculate silhouette values
        silhouette_scores_arr = []
        for cluster_num in range(2, num_clusters_total + 1):
            cluster_labels = get_clustering_labels(data_frame, cluster_num, method)
            score = silhouette_score(data_frame, cluster_labels)
            silhouette_scores_arr.append(score)
        print(silhouette_scores_arr)
        data_to_store.append(silhouette_scores_arr)

    store_path = 'output/Clustering_feature_reduction/Silhouette_for_feature_reduction_' + str(threshold) + '.csv'
    with open(store_path, 'w', newline='', encoding='utf-8-sig') as storeFile:
        csv_writer = csv.writer(storeFile)
        for i in range(len(data_to_store)):
            csv_writer.writerow(data_to_store[i])
    storeFile.close()


if __name__ == '__main__':
    # The ratio of important attributes that each paper keeps
    threshold = 0.25

    # Clustering method: Hierarchical, Hierarchical_ward, Kmeans, Birch, Agglomerative
    method = "Birch"
    # total number of clusters
    num_clusters_total = 5

    # Given the several dimension as input
    given_columns = ['Application', 'Problem', 'Metric', 'Technology', 'Blockchain', 'Consensus',
                     'Contribution', 'Evaluation', 'Security', 'Privacy', 'Communication', 'Methodology', 'AI Method',
                     'UNSD Code', 'Allocation', 'Permission', 'Type', 'Chain', 'Reward', 'TRLE']

    main()
