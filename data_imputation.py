import pandas as pd
import numpy as np

# načíta sa súbor do dataframe, hodnoty ako float, prvý stĺpec (číslo záznamu 1-300) sa dropol
def get_data_matrix(filepath):
    data = pd.read_csv(filepath)
    data = data.drop(data.columns[0], axis=1)
    data_matrix = data.to_numpy(dtype=float)

    return data_matrix

def get_mean_vector(matrix):
    col_sums = matrix.T @ np.ones([matrix.shape[0], 1])
    col_means = col_sums / matrix.shape[0]

    return col_means

def get_std_vector(matrix, mean_vector):
    std_matrix = matrix - mean_vector.T
    std_matrix *= std_matrix
    col_sums = std_matrix.T @ np.ones([std_matrix.shape[0], 1])
    col_sums /= std_matrix.shape[0]
    std_vector = np.sqrt(col_sums)

    return std_vector

# matica z matice dát sa získajú indexy s NaN a na ich miesto sa zároveň dajú priemery daného stĺpca záznamu
def init_data_matrix(matrix):
    feature_means = np.nanmean(matrix, axis=0)
    nan_coords = np.where(np.isnan(matrix))
    matrix[nan_coords] = np.take(feature_means, nan_coords[1])

    col_means = get_mean_vector(matrix)
    col_stds = get_std_vector(matrix, col_means)
    matrix -= col_means.T
    matrix /= col_stds.T

    return nan_coords

data_matrix =  get_data_matrix("v3_missing.csv")
nan_coords = init_data_matrix(data_matrix)