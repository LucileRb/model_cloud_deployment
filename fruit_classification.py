# Imports
import pandas as pd
from PIL import Image
import numpy as np
import io
import os

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras import Model
from pyspark.sql.functions import col, pandas_udf, PandasUDFType, element_at, split
from pyspark.sql import SparkSession
from pyspark.ml.feature import PCA
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import udf
from pyspark import SparkConf, SparkContext

# Configuration Spark pour l'optimisation mémoire
conf = (SparkConf()
        .setAppName("FruitClassification")
        .set("spark.executor.memory", "4g")
        .set("spark.driver.memory", "4g")
        .set("spark.executor.cores", "2")
        .set("spark.driver.cores", "2")
        .set("spark.executor.instances", "4")
        .set("spark.sql.shuffle.partitions", "200"))

sc = SparkContext(conf = conf)
spark = SparkSession(sc)

# Initialisation des chemins
PATH = 's3://p8-lucile-data/'
PATH_Data = PATH + '/Test'
PATH_Result = PATH + '/Results'

# Chargement des images
images = spark.read.format('binaryFile') \
  .option('pathGlobFilter', '*.jpg') \
  .option('recursiveFileLookup', 'true') \
  .load(PATH_Data)

images = images.withColumn('label', element_at(split(images['path'], '/'), -2))

# Repartitionnement des données (pour tirer parti du traitement distribué)
images = images.repartition(100)

# Chargement du modèle MobileNetV2
model = MobileNetV2(weights='imagenet', include_top=True, input_shape=(224, 224, 3))
new_model = Model(inputs=model.input, outputs=model.layers[-2].output)
brodcast_weights = sc.broadcast(new_model.get_weights())

def model_fn():
    model = MobileNetV2(weights='imagenet', include_top=True, input_shape=(224, 224, 3))
    for layer in model.layers:
        layer.trainable = False
    new_model = Model(inputs=model.input, outputs=model.layers[-2].output)
    new_model.set_weights(brodcast_weights.value)
    return new_model

def preprocess(content):
    img = Image.open(io.BytesIO(content)).resize([224, 224])
    arr = img_to_array(img)
    return preprocess_input(arr)

def featurize_series(model, content_series):
    input = np.stack(content_series.map(preprocess))
    preds = model.predict(input)
    output = [p.flatten() for p in preds]
    return pd.Series(output)

@pandas_udf('array<float>', PandasUDFType.SCALAR_ITER)
def featurize_udf(content_series_iter):
    model = model_fn()
    for content_series in content_series_iter:
        yield featurize_series(model, content_series)

# Application de la featurization
features_df = images.select(
    col('path'),
    col('label'),
    featurize_udf('content').alias('features')
)

# Conversion des caractéristiques en vecteurs
to_vector_udf = udf(lambda features: Vectors.dense(features), VectorUDT())
features_vector_df = features_df.withColumn('features_vector', to_vector_udf('features'))

# Détermination du k optimal
variance_df = []
for k in range(1, 101):
    pca = PCA(k=k, inputCol='features_vector', outputCol='pca_features')
    model = pca.fit(features_vector_df)
    explained_variance = model.explainedVariance.sum()
    variance_df.append((k, explained_variance))

# Convertir en DataFrame pour trouver le k optimal
variance_spark_df = spark.createDataFrame(variance_df, ['k', 'explained_variance'])
optimal_k = variance_spark_df.orderBy(variance_spark_df.explained_variance.desc()).first()[0]

print(f'Optimal number of components: {optimal_k}')

# Appliquer l'ACP avec le k optimal
pca = PCA(k=optimal_k, inputCol='features_vector', outputCol='pca_features')
pca_model = pca.fit(features_vector_df)
pca_result_df = pca_model.transform(features_vector_df)

# Sélection des colonnes nécessaires
result_df = pca_result_df.select('path', 'label', 'pca_features')

# Enregistrement des résultats réduits
result_df.write.mode('overwrite').parquet(PATH_Result)


########## END ##########