import os.path

from matplotlib import pyplot as plt
from prince import MCA, PCA
from sklearn.cluster import *
from DimensiontoBurtMatrix import *
from FeatureReduction import get_clustering_labels


def main():
    dimension_path = 'output/MCA/dimensions.csv'
    dimension1_file = pd.read_csv(dimension_path)

    data_to_store = []
    centroids_to_store = []
    for l in range(latent_dim_num):
        feature_name = f'DIM_{l} Features'
        attributes_array = dimension1_file[feature_name].to_numpy()

        for attr_num in attributes_keep_nums:
            attr_keep_list = attributes_array[:attr_num].tolist()

            # find the paper in the dataframe with these attributes
            attributes = []
            for column in df.columns:
                for attr in attr_keep_list:
                    if column in attr and column not in attributes:
                        attributes.append(column)
            selected_df = df[attributes]

            # Clustering and Calculate silhouette values
            scores_arr = []
            cluster_centroids_arr = []
            for cluster_num in range(2, MAX_CLUSTERS_NUN + 1):
                cluster_labels = get_clustering_labels(selected_df, cluster_num, clustering_method)
                if score_type == 'silhouette':
                    score = silhouette_score(selected_df, cluster_labels)
                elif score_type == 'calinski':
                    score = calinski_harabasz_score(selected_df, cluster_labels)
                elif score_type == 'davis':
                    score = davies_bouldin_score(selected_df, cluster_labels)
                else:
                    score = 0
                scores_arr.append(score)

                # output the clustering centers
                df_cluster = selected_df.copy()
                df_cluster['label'] = cluster_labels
                cluster_centroids = df_cluster.groupby('label').mean()
                cluster_centroids_test = cluster_centroids.apply(lambda x: x.apply(value_normalize))
                centroids_info_row = ''
                for cluster_id, row in cluster_centroids_test.iterrows():
                    features_with_1 = row[row == 1].index.tolist()
                    if clustering_method == 'Birch':
                        cluster_id += 1
                    cluster_centroid_info = f"Centroid {cluster_id}: {features_with_1}"
                    centroids_info_row = centroids_info_row + '; ' + cluster_centroid_info
                cluster_centroids_arr.append(centroids_info_row)

            print(scores_arr)
            data_to_store.append(scores_arr)
            centroids_to_store.append(cluster_centroids_arr)

    store_path = f'output/MCA/dimension_{clustering_method}_{score_type}_scores.csv'
    with open(store_path, 'w', newline='', encoding='utf-8-sig') as storeFile:
        csv_writer = csv.writer(storeFile)
        for i in range(len(data_to_store)):
            csv_writer.writerow(data_to_store[i])
    storeFile.close()

    store_path_centroids = f'output/MCA/dimension_{clustering_method}_{score_type}_centroids.csv'
    with open(store_path_centroids, 'w', newline='', encoding='utf-8-sig') as storeCentroidFile:
        csv_writer = csv.writer(storeCentroidFile)
        for i in range(len(centroids_to_store)):
            csv_writer.writerow(centroids_to_store[i])
    storeCentroidFile.close()


def generate_dimension_csv():
    given_columns = dimension_file['1'].dropna().values

    attributes = []
    for column in given_columns:
        attributes = attributes + [attr for attr in df.columns if column in attr]
    df_dim = df[attributes]

    # Perform MCA
    method = MCA(n_components=LATENT_DIMENSION_NUM)
    method.fit(df_dim)

    # Calculate Greenacre contributions for columns (variables)
    greenacre_contributions_cols = method.column_contributions_

    # Rank the features for each latent dimension
    output = np.array([[j for j in range(greenacre_contributions_cols.shape[0])]]).T
    for latent_id in range(4):
        contributions_one_dim = greenacre_contributions_cols[latent_id]
        ranked_contributions = contributions_one_dim.sort_values(ascending=False)
        ranked_contributions = ranked_contributions.reset_index().to_numpy()
        output = np.concatenate((output, ranked_contributions), axis=1)

    output_columns = [f'Dimension {d + 1}', 'DIM_0 Features', 'DIM_0 Values', 'DIM_1 Features', 'DIM_1 Values',
                      'DIM_2 Features', 'DIM_2 Values', 'DIM_3 Features', 'DIM_3 Values']
    output_df = pd.DataFrame(data=output, columns=output_columns)
    # write the output data frame
    path = 'output/MCA/dimensions' + '.csv'
    output_df.to_csv(path, index=False)


if __name__ == '__main__':
    # The total number of latent MCA dimensions
    latent_dim_num = 4
    # The number of top important attributes in latent dimension to keep
    attributes_keep_nums = [166, 120, 100, 80, 60, 50, 40, 30, 20]
    # Clustering evaluation method: silhouette, calinski, davis
    score_type = 'silhouette'
    # Clustering method: Hierarchical, Hierarchical_ward, Kmeans, Birch, Agglomerative
    clustering_method = 'Birch'

    important_feature_keep_ratio = 0.25
    matrix_path = f'output/Clustering_feature_reduction/binary_matrix_{important_feature_keep_ratio}.csv'
    df = pd.read_csv(matrix_path)

    # Acquire the dimension
    dimension_file_path = 'output/Reduced_dimension.csv'
    dimension_file = pd.read_csv(dimension_file_path)

    # To generate important attributes (features) of each latent MCA dimension
    generate_dimension_csv()
    # select top important attributes for clustering, and output the evaluation scores and cluster centroids
    main()
