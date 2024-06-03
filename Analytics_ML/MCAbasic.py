import os.path

from matplotlib import pyplot as plt
from prince import MCA, PCA
from sklearn.cluster import *
from DimensiontoBurtMatrix import *


def main(method_name=None, dimensions_id=None):
    output_path = 'output/' + method_name + '/'
    given_columns = dimension_file[str(dimensions_id)].dropna().values

    # 1. Sample data (replace this with your own data)
    print("Generating the data...")
    # data = data_to_dict_generate()
    df = pd.read_csv(matrix_path)
    attributes = []
    for column in given_columns:
        attributes = attributes + [attr for attr in df.columns if column in attr]
    df = df[attributes]

    # 2. Perform MCA
    method = MCA(n_components=latent_dim_num)
    method.fit(df)

    # 3. Access the results
    # Calculate the eigenvalues
    print("Eigenvalues:")
    print(method.eigenvalues_)
    print(sum(method.eigenvalues_))
    benzecri_values = method.eigenvalues_ / sum(method.eigenvalues_)
    ax = plt.subplot()
    x = [d for d in range(1, len(method.eigenvalues_) + 1)]
    ax.plot(x, method.eigenvalues_, 'o', color='k')
    # ax.plot(x, benzecri_values, linestyle='--', color='r', label='Benzécri')
    ax.legend(fontsize=20)
    ax.set_xlabel(method_name + ' Dimension', fontsize=24)
    ax.set_ylabel('Eigenvalues', fontsize=24)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    output_fig_name = output_path + method_name + '_dimensions_eigenvalues.pdf'
    # plt.xticks(x)
    plt.savefig(output_fig_name, dpi=300, bbox_inches='tight')
    plt.show()
    eigen_output = output_path + 'EigenValues.csv'
    with open(eigen_output, 'w', newline='', encoding='utf-8-sig') as outFile:
        csv_writer = csv.writer(outFile)
        csv_writer.writerow(['Latent Dimension', 'Eigen Value', 'Benzécri'])
        for i in range(len(method.eigenvalues_)):
            line = [i + 1, method.eigenvalues_[i], round(benzecri_values[i], 5)]
            csv_writer.writerow(line)
    outFile.close()
    # 3+. Calculate Benzécri contributions for rows (categories)
    benzecri_contributions_rows = method.row_contributions_
    print("Benzécri:")
    print(benzecri_contributions_rows)
    # Calculate Greenacre contributions for columns (variables)
    greenacre_contributions_cols = method.column_contributions_
    print("Greenacre:")
    print(greenacre_contributions_cols)

    # 4. Output the Contribution of each latent MCA/PCA dimension
    output_file = output_path + 'Contribution_' + method_name + '_dimensions.csv'
    with open(output_file, 'w', newline='', encoding='utf-8-sig') as outFile:
        csv_writer = csv.writer(outFile)
        first_row = greenacre_contributions_cols[0].index
        first_row = [title.replace(".0", "") for title in first_row]
        csv_writer.writerow(first_row)
        for i in range(len(method.eigenvalues_)):
            line = greenacre_contributions_cols[i].values
            csv_writer.writerow(line)
    outFile.close()

    # 5. Plot the contribution of features in 4 most important latent MCA/PCA dimension
    input_data = greenacre_contributions_cols.transpose()
    features_names_splits = input_data.columns.str.split('_')
    features_names = features_names_splits.str[0] + '_' + features_names_splits.str[1]
    df_features_importance = input_data.groupby(features_names, axis=1).sum()
    for i in range(4):
        features_cont = df_features_importance.iloc[i]
        top_10_features = features_cont.nlargest(10)
        x = top_10_features.index
        y = top_10_features.values
        # Plot the figure
        plt.figure(figsize=(8, 6))
        # plt.barh(x, y, height=0.3, color='LightSkyBlue')
        plt.bar(x, y, width=0.3, color='b')
        plt.ylabel('Contribution (%)', fontsize=24)
        plt.xticks(fontsize=18, rotation=90)
        plt.yticks(fontsize=16)
        figure_name = method_name + '-dimension-' + str(i + 1) + '.pdf'
        plt.savefig(output_path + figure_name, dpi=300, bbox_inches='tight')
        plt.show()

    # 6. Calculate the sum variance of each dimension
    # Get the importance/contribution of each old dimension to latent dimension
    dimension_names = input_data.columns.str.split('_').str[0]
    df_dimension_importance = input_data.groupby(dimension_names, axis=1).sum()
    df_dimension_importance_store = df_dimension_importance.copy()
    df_dimension_importance = df_dimension_importance.apply(np.sum, axis=0)  # features transform to index
    # Rank these dimensions
    ranked_series = df_dimension_importance.rank(method='min', ascending=False)
    df_dimension_importance = pd.concat([df_dimension_importance, ranked_series], axis=1)
    df_dimension_importance.columns = ['Importance', 'Rank']
    df_dimension_importance = df_dimension_importance.sort_values(by='Rank')
    # Draw the plot
    names = df_dimension_importance['Importance'].index
    values = df_dimension_importance['Importance'].values
    values = values / sum(values)
    plt.plot(names, values, 'o-k')
    plt.yticks(fontsize=18)
    plt.xticks(rotation=90, fontsize=20)
    plt.xlabel('Design Dimension', fontsize=24)
    plt.ylabel('Eigenvalue', fontsize=24)
    # for i in range(len(keys)):
    #     plt.text(keys[i], values[i], '{:.2f}'.format(values[i]), ha='center', va='bottom', fontsize=16)
    plt.savefig(output_path + 'MCA_dimension_variance_rank.pdf', dpi=300, bbox_inches='tight')
    plt.show()

    # 7. Store the importance of features for MCA dimensions as input data to chord diagram
    chord_data = {'Source area': [], 'Target area': [], 'Millions of persons': []}
    chord_data = pd.DataFrame(chord_data)
    for mca_idx in range(20):
        feature_values_in_mca = df_dimension_importance_store.iloc[mca_idx]
        features_names = feature_values_in_mca.index
        for feature_idx in range(len(feature_values_in_mca)):
            new_data = {'Source area': [f'MCA {mca_idx + 1}'],
                        'Target area': [features_names[feature_idx]],
                        'Millions of persons': [feature_values_in_mca.iloc[feature_idx]]
                        }
            new_data = pd.DataFrame(new_data)
            chord_data = pd.concat([chord_data, new_data], ignore_index=True)
    chord_data_path = 'output/chord_diagram.csv'
    chord_data.to_csv(chord_data_path, index=False)


if __name__ == '__main__':
    latent_dim_num = 100

    important_feature_keep_ratio = 0.25
    matrix_path = f'output/Clustering_feature_reduction/binary_matrix_{important_feature_keep_ratio}.csv'

    # Acquire the dimension
    dimension_file_path = 'output/Reduced_dimension.csv'
    dimension_file = pd.read_csv(dimension_file_path)
    given_index = 1
    main('MCA', given_index)
