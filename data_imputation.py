import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

# načíta sa súbor do dataframe, hodnoty ako float, prvý stĺpec (číslo záznamu 1-300) sa dropol
def get_data_matrix(filepath):
    data = pd.read_csv(filepath)
    data = data.drop(data.columns[0], axis=1)
    data_matrix = data.to_numpy(dtype=float)

    return data_matrix

# matica z matice dát sa získajú indexy s NaN a na ich miesto sa zároveň dajú priemery daného stĺpca záznamu
# vráti indexy pre kde boli chýbajúce NaN hodnoty
def init_data_matrix(matrix):
    feature_means = np.nanmean(matrix, axis=0)
    nan_coords = np.where(np.isnan(matrix))
    valid_coords = np.where(~np.isnan(matrix))
    matrix[nan_coords] = np.take(feature_means, nan_coords[1])

    return nan_coords, valid_coords

# potom sa zavolá už s STD maticou v hlavnom cykle - UPRAVIŤ !!!!
def compute_PCA(std_matrix):
    cov_matrix = 1/(std_matrix.shape[0]-1) * ( std_matrix.T @ std_matrix)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix) # miesto eig, eigh je pre symetrické matice
    sorted_desc = np.argsort(eigenvalues)[::-1]  # zoradenie zostupne - aby prvý komponent mal najvyšší variance
    eigenvectors = eigenvectors[:, sorted_desc]
    
    return eigenvectors

def get_imputated_matrix(std_data_matrix, eigenvectors, n_components):
    loadings_matrix = eigenvectors[:, :n_components]
    scores_matrix = std_data_matrix @ loadings_matrix
    reconstructed_matrix = scores_matrix @ loadings_matrix.T

    return reconstructed_matrix

def get_objective(original_matrix, imputated_matrix, nan_coords):
    objective_matrix = original_matrix - imputated_matrix
    objective_matrix[nan_coords] = 0
    objective = np.sum(objective_matrix**2)
    
    return objective

data_matrix =  get_data_matrix("v3_missing.csv")
nan_coords, valid_coords = init_data_matrix(data_matrix)
std_matrix = StandardScaler().fit_transform(data_matrix)
eigenvectors = compute_PCA(std_matrix)
imputated_matrix = get_imputated_matrix(std_matrix, eigenvectors, 1)
data_matrix[nan_coords] = imputated_matrix[nan_coords]
objective = get_objective(data_matrix, imputated_matrix, nan_coords)