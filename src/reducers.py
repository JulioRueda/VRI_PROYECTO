from sklearn.decomposition import PCA 
from sklearn.decomposition import KernelPCA as KPCA 
import pandas as pd 
import os 
import umap 
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

class pca():
    name = None

    def __init__(self, name):
        self.name = name

    def reducir_dimension(self,df_train, df_test, columna, reduccion):
        np.random.seed(42)
        # Extraer embeddings de las columnas especificadas
        embeddings_train = df_train[columna].tolist()
        embeddings_test = df_test[columna].tolist()
        
        # Crear y entrenar el modelo PCA con los datos de entrenamiento
        pca = PCA(n_components=reduccion)
        reduced_embeddings_train = pca.fit_transform(embeddings_train)
        
        # Aplicar la transformación PCA a los datos de prueba
        reduced_embeddings_test = pca.transform(embeddings_test)
        
        return reduced_embeddings_train, reduced_embeddings_test



class umapp():
    name = None

    def __init__(self, name):
        self.name = name

    def reducir_dimension(self,df_train, df_test, columna, reduccion):
        np.random.seed(42)
        # Extraer embeddings de las columnas especificadas
        embeddings_train = df_train[columna].tolist()
        embeddings_test = df_test[columna].tolist()
        n_train = len(embeddings_train)
        embeddings_ = embeddings_train + embeddings_test
        # invocamos a UMAP
        umap_ = umap.UMAP(n_components=reduccion)
        embeddings_ = umap_.fit_transform(np.vstack(embeddings_))

        return embeddings_[0:n_train], embeddings_[n_train:]

class kpca():
    name = None

    def __init__(self, name):
        self.name = name

    def reducir_dimension(self, df_train, df_test, columna, reduccion):
        np.random.seed(42)
        # Extraer embeddings de las columnas especificadas
        embeddings_train = df_train[columna].tolist()
        embeddings_test = df_test[columna].tolist()
        
        # Crear y entrenar el modelo PCA con los datos de entrenamiento
        pca = KPCA(n_components=reduccion, kernel='rbf')
        reduced_embeddings_train = pca.fit_transform(embeddings_train)
        
        # Aplicar la transformación PCA a los datos de prueba
        reduced_embeddings_test = pca.transform(embeddings_test)
        
        return reduced_embeddings_train, reduced_embeddings_test


class ae():
    name = None

    def __init__(self, name):
        self.name = name

    def mi_autoencoder(self,dim_reduction=100, activation_func='tanh', output_func='sigmoid'):
        np.random.seed(42)
        #Input Layer
        input_layer = tf.keras.Input(shape=(768,))

        #Encoder layers
        encoded = layers.Dense(768, activation=activation_func)(input_layer) 
        encoded = layers.Dense(dim_reduction,activation = activation_func)(encoded)

        #Decoder layers 
        decoded = layers.Dense(768, activation = activation_func)(encoded) 
        decoded = layers.Dense(768, activation=output_func)(decoded) 

        #Autoencoder en modelo
        autoencoder = Model(inputs=input_layer, outputs = decoded) 

        # Modelo codificador para reducción de dimensión
        encoder = Model(inputs=input_layer, outputs=encoded)

        #Compilar, cosine similarity porque son embeddings
        autoencoder.compile(optimizer = 'adam', loss = 'mean_squared_error') 

        return autoencoder, encoder

    def reducir_dimension(self,df_train, df_test, columna, reduccion):
        np.random.seed(42)
        # Extraer embeddings de las columnas especificadas
        embeddings_train = np.vstack(df_train[columna])
        embeddings_test = np.vstack(df_test[columna])
        
        autoencoder,encoder = mi_autoencoder(dim_reduction=reduccion)
        autoencoder.fit(
            embeddings_train,embeddings_train,
            epochs=10,
            batch_size=64,
            verbose=0
        )
        reduced_train = encoder.predict(embeddings_train)
        reduced_test  = encoder.predict(embeddings_test)
        return reduced_train,reduced_test


def get_reducer(reducer_name):
    if reducer_name== 'pca':
        return pca(reducer_name)
    elif reducer_name == 'kpca':
        return kpca(reducer_name)
    elif reducer_name == 'umap':
        return umapp(reducer_name)
    elif reducer_name == 'ae':
        return ae(reducer_name)

