import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

class PCAImputation:
    def __init__(self, filepath):
        self.data_matrix = self.get_data_matrix(filepath)
        self.nan_coords, self.valid_coords = self.init_data_matrix(self.data_matrix)
        
    # načíta sa súbor do dataframe, hodnoty ako float, prvý stĺpec (číslo záznamu 1-300) sa dropol
    def get_data_matrix(self, filepath):
        data = pd.read_csv(filepath)
        data = data.drop(data.columns[0], axis=1)
        data_matrix = data.to_numpy(dtype=float)
        return data_matrix

    # matica z matice dát sa získajú indexy s NaN a na ich miesto sa zároveň dajú priemery daného stĺpca záznamu
    # vráti indexy pre kde boli chýbajúce NaN hodnoty
    def init_data_matrix(self):
        feature_means = np.nanmean(self.data_matrix, axis=0)
        nan_coords = np.where(np.isnan(self.data_matrix))
        valid_coords = np.where(~np.isnan(self.data_matrix))
        self.data_matrix[nan_coords] = np.take(feature_means, nan_coords[1])

        return nan_coords, valid_coords

    @staticmethod
    def compute_PCA(std_matrix):
        cov_matrix = 1/(std_matrix.shape[0]-1) * ( std_matrix.T @ std_matrix)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix) # miesto eig, eigh je pre symetrické matice
        sorted_desc = np.argsort(eigenvalues)[::-1]  # zoradenie zostupne - aby prvý komponent mal najvyšší variance
        eigenvectors = eigenvectors[:, sorted_desc]
    
        return eigenvectors

    @staticmethod
    def get_imputated_matrix(std_data_matrix, eigenvectors, n_components):
        loadings_matrix = eigenvectors[:, :n_components]
        scores_matrix = std_data_matrix @ loadings_matrix
        reconstructed_matrix = scores_matrix @ loadings_matrix.T

        return reconstructed_matrix
    
    @staticmethod
    def get_objective(original_matrix, imputated_matrix, nan_coords):
        objective_matrix = original_matrix - imputated_matrix
        objective_matrix[nan_coords] = 0
        objective = np.sum(objective_matrix**2)
    
        return objective
    
    # hlavná funkcia, algoritmus na imputation s PCA
    def imputate_data(self, n_components):
        current_objective = -1
        new_objective = -1
        
        while (True):
            std_matrix = StandardScaler().fit_transform(self.data_matrix)
            eigenvectors = self.compute_PCA(std_matrix)
            imputated_matrix = self.get_imputated_matrix(std_matrix, eigenvectors, n_components)
            self.data_matrix[self.nan_coords] = imputated_matrix[self.nan_coords]
            new_objective = self.get_objective(self.data_matrix, imputated_matrix, self.nan_coords)
            
            # pokiaľ sa riešenie zlepšilo, nastaví sa nová hodnota a pokračuje sa
            if new_objective < current_objective or current_objective == -1:
                current_objective = new_objective
            else:
                break