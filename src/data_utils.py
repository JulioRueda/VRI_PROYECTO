import pandas as pd 
import numpy as np
import yaml
import os

def change_format(df):
    df['Embedding_Respuesta'] = df['Embedding_Respuesta'].apply(
        lambda x: np.fromstring(x.replace("[", "").replace("]", "").replace("\n", ""), sep=' '))
    df['Embedding_Concat1'] = df['Embedding_Concat1'].apply(
        lambda x: np.fromstring(x.replace("[", "").replace("]", "").replace("\n", ""), sep=' '))
    df['Embedding_Concat2'] = df['Embedding_Concat2'].apply(
        lambda x: np.fromstring(x.replace("[", "").replace("]", "").replace("\n", ""), sep=' '))
    
    return df 

# def load_config(path='G:\Mi unidad\Mi CIUP\LOS REALES CODIGOS\VRI\config\config.yaml'):
#     with open(path, "r") as f:
#         return yaml.safe_load(f)

def find_file(direction):
    return os.path.isfile(direction)