import PCA_imputation as pcai

pca_imputation = pcai.PCA_imputation('v3_missing.csv', 'v3_complete.csv')
final_error, rmse, mae = pca_imputation.imputate_data(n_components=1, evaluate=True)
print(f"Imputation using 1 component final error: {final_error}, RMSE: {rmse}, MAE: {mae}")

pca_imputation = pcai.PCA_imputation('v3_missing.csv', 'v3_complete.csv')
final_error, rmse, mae = pca_imputation.imputate_data(n_components=2, evaluate=True)
print(f"Imputation using 2 components final error: {final_error}, RMSE: {rmse}, MAE: {mae}")
pca_imputation.run_experiments(n_components=2, replics=1000, missing_values=183)
pca_imputation.run_experiments(n_components=2, replics=1000, missing_values=250)
pca_imputation.run_experiments(n_components=2, replics=1000, missing_values=500)

pca_imputation = pcai.PCA_imputation('v3_missing.csv', 'v3_complete.csv')
final_error, rmse, mae = pca_imputation.imputate_data(n_components=3, max_iter=1000, evaluate=True)
print(f"Imputation using 3 components final error: {final_error}, RMSE: {rmse}, MAE: {mae}") # dobré na experiment - od nejakých 5000 vyššie sa veľmi spomaľuje zlepšovanie