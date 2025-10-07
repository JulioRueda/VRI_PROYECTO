from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import MinMaxScaler
from reducers import get_reducer 
from models import get_model 
import joblib
import numpy as np
import os 
from evaluate import evaluate
import mlflow
import hydra
import pandas as pd 
from data_utils import change_format, find_file
from omegaconf import DictConfig
import warnings 
warnings.filterwarnings('ignore')







@hydra.main(config_path="../conf", config_name='config', version_base=None)
def run_experiment(cfg: DictConfig):

    #SETTINGS OF MLFLOW
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    
    print("MLFLOW registrado")

    COLUMNS = ['Embedding_Respuesta','Embedding_Concat1', 'Embedding_Concat2']

    #REDUCTION
    reducer = get_reducer(cfg.reducer.name)
    file_reduced_T = rf"G:\Mi unidad\Mi CIUP\LOS REALES CODIGOS\VRI\data\reduced\{cfg.dataset.name}_{cfg.experiment.seed}_{cfg.reducer.name}_Train_N.xlsx"
    file_reduced_t = rf"G:\Mi unidad\Mi CIUP\LOS REALES CODIGOS\VRI\data\reduced\{cfg.dataset.name}_{cfg.experiment.seed}_{cfg.reducer.name}_Test_N.xlsx"

    file_normalized_T = rf"G:\Mi unidad\Mi CIUP\LOS REALES CODIGOS\VRI\data\processed_trials\normalized_trials\{cfg.dataset.name}_{cfg.experiment.seed}_Train_N.xlsx"
    file_normalized_t = rf"G:\Mi unidad\Mi CIUP\LOS REALES CODIGOS\VRI\data\processed_trials\normalized_trials\{cfg.dataset.name}_{cfg.experiment.seed}_Test_N.xlsx"

    file__T = rf"G:\Mi unidad\Mi CIUP\LOS REALES CODIGOS\VRI\data\processed_trials\{cfg.dataset.name}_{cfg.experiment.seed}_Train.xlsx"
    file__t = rf"G:\Mi unidad\Mi CIUP\LOS REALES CODIGOS\VRI\data\processed_trials\{cfg.dataset.name}_{cfg.experiment.seed}_Test.xlsx"

    if find_file(file_reduced_T):
        print('caso optimo')
        #Optimal case
        df_T = pd.read_excel(file_reduced_T)
        df_t = pd.read_excel(file_reduced_T)
        df_T = change_format(df_T) 
        df_t = change_format(df_t)
        for columna in COLUMNS:
            #PARECE QUE ESTO NO ESTÁ FUNCIONANDO, REVISALO CON CALMA
            print('ingresó al for columna')
            print(rf"{columna}_{cfg.reducer.name}_{cfg.reducer.params.n_components}" not in df_T.columns)
            if rf"{columna}_{cfg.reducer.name}_{cfg.reducer.params.n_components}" not in df_T.columns:
                print('ingresó al if')
                reduced_train, reduced_test = reducer.reducir_dimension(df_T,df_t,columna,cfg.reducer.params.n_components)
                df_T[rf"{columna}_{cfg.reducer.name}_{cfg.reducer.params.n_components}"] = list(reduced_train)
                df_t[rf"{columna}_{cfg.reducer.name}_{cfg.reducer.params.n_components}"] = list(reduced_test)
                
                df_T.to_excel(file_reduced_T)
                df_t.to_excel(file_reduced_t)
            else:
                print('ingresó al else')
                df_T = change_format(df_T, columna= rf"{columna}_{cfg.reducer.name}_{cfg.reducer.params.n_components}")
                df_t = change_format(df_t, columna= rf"{columna}_{cfg.reducer.name}_{cfg.reducer.params.n_components}")

    else:
        if find_file(file_normalized_T):
            print('caso normalizado')
            df_T = pd.read_excel(file_normalized_T)
            df_t = pd.read_excel(file_normalized_t)
            print('archivos leidos')
            df_T = change_format(df_T) 
            df_t = change_format(df_t)
            print('archivos modificados')
            for columna in COLUMNS:
                reduced_train, reduced_test = reducer.reducir_dimension(df_T,df_t,columna,cfg.reducer.params.n_components)
            
                df_T[rf"{columna}_{cfg.reducer.name}_{cfg.reducer.params.n_components}"] = list(reduced_train)
                df_t[rf"{columna}_{cfg.reducer.name}_{cfg.reducer.params.n_components}"] = list(reduced_test)
            print('archivos reducidos')
            df_T.to_excel(file_reduced_T)
            df_t.to_excel(file_reduced_t)
            print('sali del caso normalizado')

        else:
            if find_file(file__T):
                print('caso trializado')
                df_T = pd.read_excel(file__T) 
                df_t = pd.read_excel(file__t) 
                df_T = change_format(df_T) 
                df_t = change_format(df_t)
                for col in COLUMNS:
                    scaler = MinMaxScaler()
                    df_T[col] = list(scaler.fit_transform(df_T[col].tolist()))
                    df_t[col] = list(scaler.transform(df_t[col].tolist()))
                df_T.to_excel(file_normalized_T)
                df_t.to_excel(file_normalized_t)
                #reducir
                for columna in COLUMNS:
                    reduced_train, reduced_test = reducer.reducir_dimension(df_T,df_t,columna,cfg.reducer.params.n_components)
                    df_T[rf"{columna}_{cfg.reducer.name}_{cfg.reducer.params.n_components}"] = list(reduced_train)
                    df_t[rf"{columna}_{cfg.reducer.name}_{cfg.reducer.params.n_components}"] = list(reduced_test)
                
                df_T.to_excel(file_reduced_T)
                df_t.to_excel(file_reduced_t)
            else:
                print('caso nulo')
                df = pd.read_excel(cfg.dataset.path)
                df = change_format(df)
                df_T, df_t = train_test_split(df,test_size=cfg.experiment.test_size, random_state=cfg.experiment.seed)
                df_T.to_excel(file__T)
                df_t.to_excel(file__t)
                #normalize
                for col in COLUMNS:
                    scaler = MinMaxScaler()
                    df_T[col] = list(scaler.fit_transform(df_T[col].tolist()))
                    df_t[col] = list(scaler.transform(df_t[col].tolist()))
                df_T.to_excel(file_normalized_T)
                df_t.to_excel(file_normalized_t)
                #reducir
                for columna in COLUMNS:
                    reduced_train, reduced_test = reducer.reducir_dimension(df_T,df_t,columna,cfg.reducer.params.n_components)
                    df_T[rf"{columna}_{cfg.reducer.name}_{cfg.reducer.params.n_components}"] = list(reduced_train)
                    df_t[rf"{columna}_{cfg.reducer.name}_{cfg.reducer.params.n_components}"] = list(reduced_test)
                df_T.to_excel(file_reduced_T)
                df_t.to_excel(file_reduced_t)

    print('- columnas cargadas')

    for columna in COLUMNS:
        print(RF'----->   {columna}')
        
        #MACHINE LEARNING
        model = get_model(name = cfg.model.name)
        with mlflow.start_run(
            run_name=f"{cfg.experiment.seed}_{columna}_{cfg.reducer.name}_{cfg.reducer.params.n_components}_{cfg.model.name}"

        ):
            model.fit(np.vstack(df_T[rf"{columna}_{cfg.reducer.name}_{cfg.reducer.params.n_components}"]), df_T[cfg.dataset.target])

            preds = model.predict(np.vstack(df_t[rf"{columna}_{cfg.reducer.name}_{cfg.reducer.params.n_components}"]))
            metrics = evaluate(df_t[cfg.dataset.target], preds)
            

            #LOG PARAMS, MODELS, METRICS 
            mlflow.log_param('trial', cfg.experiment.seed)
            mlflow.log_param('reducer', cfg.reducer.name)
            mlflow.log_param('model', cfg.model.name)
            mlflow.log_param('column', columna)
            mlflow.log_param('n_dim', cfg.reducer.params.n_components) 

            for k,v in metrics.items():
                mlflow.log_metric(k,v) 
            
            mlflow.sklearn.log_model(model, name='model')

            mlflow.end_run()
    

if __name__=="__main__":
    run_experiment()