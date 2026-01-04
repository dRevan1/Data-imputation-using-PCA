import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

class PCA_imputation:
    def __init__(self, filepath_missing, filepath_complete=None):
        self.filepath_missing = filepath_missing
        self.filepath_complete = filepath_complete
        self.load_files()
        
    
    def load_files(self):
        self.data_matrix = self.get_data_matrix(self.filepath_missing)
        if self.filepath_complete:
            self.complete_data_matrix = self.get_data_matrix(self.filepath_complete)  # pre complete data set na experimenty, my ho máme
            self.complete_std_matrix = StandardScaler().fit(self.complete_data_matrix).transform(self.complete_data_matrix)
        else:
            self.complete_data_matrix, self.complete_std_matrix = None, None
            
        self.nan_coords, self.valid_coords = self.init_data_matrix()
        self.missing_std_matrix = StandardScaler().fit(self.data_matrix).transform(self.data_matrix)
        self.missing_mean_ = StandardScaler().fit(self.data_matrix).mean_
        self.missing_scale_ = StandardScaler().fit(self.data_matrix).scale_
     
            
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


    def compute_PCA(self):
        cov_matrix = 1/(self.missing_std_matrix.shape[0]-1) * ( self.missing_std_matrix.T @ self.missing_std_matrix)
        _, eigenvectors = np.linalg.eigh(cov_matrix) # miesto eig, eigh je pre symetrické matice
        eigenvectors = eigenvectors[:, ::-1] # zoradenie zostupne - aby prvý komponent mal najvyšší variance
    
        return eigenvectors


    def get_imputated_matrix(self, eigenvectors, n_components):
        loadings_matrix = eigenvectors[:, :n_components] # vyberie sa n_components vektorov - hlavných komponentov
        scores_matrix = self.missing_std_matrix @ loadings_matrix # projekcia dát na hlavné komponenty
        reconstructed_matrix = scores_matrix @ loadings_matrix.T # rekonštrukcia dát z hlavných komponentov

        return reconstructed_matrix
    
    
    # účelová funkcia, cieľová hodnota, ktorá sa počíta s každou iteráciou algoritmu, aby sme zistili, či sa riešenie zlepšilo
    # počítame ako rozdiel medzi dátami z pôvodnej matice a rekonštruovanej matice pre hodnoty, ktoré neboli NaN
    @staticmethod
    def get_objective(original_matrix, imputated_matrix, nan_coords):
        objective_matrix = original_matrix - imputated_matrix
        objective_matrix[nan_coords] = 0
        objective = np.sum(objective_matrix**2)
    
        return objective
    
    
    # vráti mean absolute error a root mean square error pre doplnené hodnoty oproti matici kompletných dát
    def evaluate_imputation(self):
        if self.complete_data_matrix is None:
            print("Complete data matrix missing - evaluation cannot be performed.")
            return -1, -1
        
        error_matrix = self.complete_data_matrix - self.data_matrix
        error_matrix[self.valid_coords] = 0
        
        rmse = np.sum(error_matrix**2)
        rmse = np.sqrt(rmse / len(self.nan_coords[0]))
        
        mae = np.sum(np.abs(error_matrix))
        mae = mae / len(self.nan_coords[0])
        
        return rmse, mae
    
    
    # hlavná funkcia, algoritmus na imputation s PCA
    def imputate_data(self, n_components, evaluate=False, max_iter=1000):
        current_objective = -1
        new_objective = -1
        rmse = -1
        mae = -1
        
        for i in range(max_iter):
            eigenvectors = self.compute_PCA()
            imputated_matrix = self.get_imputated_matrix(eigenvectors, n_components)
            self.missing_std_matrix[self.nan_coords] = imputated_matrix[self.nan_coords]
            new_objective = self.get_objective(self.missing_std_matrix, imputated_matrix, self.nan_coords)
            
            # pokiaľ sa riešenie zlepšilo, nastaví sa nová hodnota a pokračuje sa
            if new_objective < current_objective or current_objective == -1:
                current_objective = new_objective
            else:
                break
            
        imputated_matrix = (imputated_matrix * self.missing_scale_) + self.missing_mean_  # denormalizácia, aby sa doplnili chýbajúce hodnoty v pôvodnej mierke    
        self.data_matrix[self.nan_coords] = imputated_matrix[self.nan_coords]
        #print(f"Converged after {i + 1} iterations.")
        
        if evaluate:
            rmse, mae = self.evaluate_imputation()
            
        return current_objective, rmse, mae
    
    
    # nastaví náhodne zadaný počet chýbajúcich údajov v kompletnej matici pre experimenty, vychádza z kompletnej matice údajov
    def set_random_missing_values(self, missing_values=183):
        self.data_matrix = self.complete_data_matrix.copy()
        self.missing_std_matrix, self.missing_mean_, self.missing_scale_ = StandardScaler().fit(self.data_matrix).transform(self.data_matrix), StandardScaler().fit(self.data_matrix).mean_, StandardScaler().fit(self.data_matrix).scale_
        
        missing_data = 0
        while missing_data < missing_values:
            rand_row = np.random.randint(0, self.data_matrix.shape[0])
            rand_col = np.random.randint(0, self.data_matrix.shape[1])
            if not np.isnan(self.data_matrix[rand_row, rand_col]):
                self.data_matrix[rand_row, rand_col] = np.nan
                missing_data += 1
                          
        self.nan_coords, self.valid_coords = self.init_data_matrix()


    # experiment so zadaným počtom replikácií, vždy sa náhodne vyberú chýbajúce hodnoty podľa kompletnej matice
    def run_experiments(self, n_components, max_iter=1000, replics=100, missing_values=183):
        if self.complete_data_matrix is None:
            print("Complete data matrix missing - experiments cannot be run.")
            return
        
        avg_error = 0
        avg_rmse = 0
        avg_mae = 0
        
        for i in range(replics):
            self.set_random_missing_values(missing_values=missing_values)
            final_error, rmse, mae = self.imputate_data(n_components=n_components, evaluate=True, max_iter=max_iter)
            avg_error += final_error
            avg_rmse += rmse
            avg_mae += mae
        avg_error /= replics
        avg_rmse /= replics
        avg_mae /= replics
        self.init_data_matrix() # reset na pôvodné matice
        
        print(f"Experiment - {replics} replics, {n_components} principal components, {missing_values} random missing values:")
        print(f"Average final objective values: {avg_error}, average RMSE: {avg_rmse}, average MAE: {avg_mae}")