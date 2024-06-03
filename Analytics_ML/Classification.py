from DimensiontoBurtMatrix import *
import matplotlib.pyplot as plt
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from scipy.stats import ttest_ind


def main():
    correlation_data = matrix.corr()

    feature_importance_output = np.zeros((len(given_columns), len(given_columns)))
    correlations_output = np.zeros((len(given_columns), len(given_columns)))
    total_attributes_in = []
    for row_idx, dimension in enumerate(given_columns):
        print(f"The dimension: {dimension}")

        # find the corresponding data
        attributes_in = [attr for attr in matrix.columns if dimension in attr]
        attributes_out = [attr for attr in matrix.columns if attr not in attributes_in]
        data_target = correlation_data.loc[attributes_out, attributes_in]
        total_attributes_in += attributes_in

        # sort the importance of data
        data_target['sum'] = data_target.sum(axis=1, skipna=True)
        data_target = data_target.sort_values(by='sum', ascending=False)
        # find the top important features
        data_target_top = data_target.iloc[:int(len(data_target))]
        important_attrs = data_target_top.index

        # get the new correlation matrix
        correlations_dim = data_target.drop(columns=['sum']).to_numpy()
        correlations_dim = np.nan_to_num(correlations_dim)

        # Multi-tree classification
        target = matrix[attributes_in]
        new_matrix = matrix[important_attrs]
        # Discrete the data into train and test
        x_train, x_test, y_train, y_test = train_test_split(np.array(new_matrix), np.array(target),
                                                            test_size=0.3, random_state=42)
        classifier = DecisionTreeClassifier(max_depth=7, min_samples_leaf=10)
        rf_model = MultiOutputClassifier(classifier)
        rf_model.fit(x_train, y_train)
        estimator = rf_model.estimators_[0]
        feature_importance = estimator.feature_importances_

        # output the feature importance and correlations
        feature_importance_arr = np.zeros(len(given_columns))
        correlations_arr = np.zeros((len(given_columns), len(attributes_in)))
        for i, attr in enumerate(important_attrs):
            for j, col in enumerate(given_columns):
                if col in attr:
                    feature_importance_arr[j] += feature_importance[i]
                    correlations_arr[j] += correlations_dim[i]

        feature_importance_output[row_idx] = feature_importance_arr
        correlations_arr = np.mean(np.transpose(correlations_arr), axis=0)
        correlations_output[row_idx] = correlations_arr

    output_data = pd.DataFrame(feature_importance_output, columns=given_columns)
    output_path = f'output/Classifier/feature_importance.csv'
    output_data.to_csv(output_path, index=False)
    output_data2 = pd.DataFrame(correlations_output, columns=given_columns, index=given_columns)
    output_path2 = f'output/Classifier/correlations.csv'
    output_data2.to_csv(output_path2)

    # calculate p-value for each pair of features
    p_values_arr = np.zeros((matrix.shape[1], matrix.shape[1]))
    for i, col_i in enumerate(matrix.columns):
        print(f"For p-value of {col_i}")
        for j, col_j in enumerate(matrix.columns):
            t_statistic, p_value = ttest_ind(matrix[col_i], matrix[col_j])
            p_values_arr[i][j] = p_value
    p_values = pd.DataFrame(p_values_arr, columns=matrix.columns, index=matrix.columns)
    # get mean of values for each dimension
    p_values_dim = p_values.copy()
    old_columns = p_values_dim.columns
    for col in given_columns:
        c_cols_to_sum = [c_col for c_col in old_columns if col in c_col]
        p_values_dim[col] = p_values_dim[c_cols_to_sum].mean(axis=1)
    p_values_dim = p_values_dim.drop(columns=old_columns)
    for col in given_columns:
        c_index_to_sum = [c_col for c_col in old_columns if col in c_col]
        p_values_dim.loc[col] = p_values_dim.loc[c_index_to_sum].mean(axis=0)
    p_values_dim = p_values_dim.drop(index=old_columns)
    print(p_values_dim)
    p_path = f'output/Classifier/p_values.csv'
    p_values_dim.to_csv(p_path)


if __name__ == '__main__':
    # Read the binary matrix for all dimensions
    matrix_path = 'output/binary_matrix.csv'
    matrix = pd.read_csv(matrix_path)
    data_matrix = np.array(matrix)

    given_columns = ['Methodology', 'Application', 'Problem', 'Contribution', 'AI Method', 'Security', 'Privacy',
                     'Allocation', 'Metric', 'Technology', 'TRLE', 'UNSD Code', 'Communication',
                     'Evaluation', 'Blockchain', 'Consensus', 'Permission', 'Type', 'Chain', 'Reward']

    # calculate the feature importance and correlation using multi-tree classification approach
    main()
