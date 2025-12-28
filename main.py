import PCA_imputation as pcai

pca_imputation = pcai.PCA_imputation('v3_missing.csv')
print(f"Imputation using 1 component final error: {pca_imputation.imputate_data(n_components=1)}")
pca_imputation = pcai.PCA_imputation('v3_missing.csv')
print(f"Imputation using 2 components final error: {pca_imputation.imputate_data(n_components=2)}")
pca_imputation = pcai.PCA_imputation('v3_missing.csv')
print(f"Imputation using 3 components final error: {pca_imputation.imputate_data(n_components=3, max_iter=1000)}") # dobré na experiment - od nejakých 5000 vyššie sa veľmi spomaľuje zlepšovanie