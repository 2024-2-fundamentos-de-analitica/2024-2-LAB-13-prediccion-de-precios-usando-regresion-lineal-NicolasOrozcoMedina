#
# En este dataset se desea pronosticar el precio de vhiculos usados. El dataset
# original contiene las siguientes columnas:
#
# - Car_Name: Nombre del vehiculo.
# - Year: Año de fabricación.
# - Selling_Price: Precio de venta.
# - Present_Price: Precio actual.
# - Driven_Kms: Kilometraje recorrido.
# - Fuel_type: Tipo de combustible.
# - Selling_Type: Tipo de vendedor.
# - Transmission: Tipo de transmisión.
# - Owner: Número de propietarios.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# pronostico están descritos a continuación.
#
#
# Paso 1.
# Preprocese los datos.
# - Cree la columna 'Age' a partir de la columna 'Year'.
#   Asuma que el año actual es 2021.
# - Elimine las columnas 'Year' y 'Car_Name'.
import pandas as pd

train = pd.read_csv('files/input/train_data.csv')
test = pd.read_csv('files/input/test_data.csv')
train['Age'] = 2021 - train['Year']
test['Age'] = 2021 - test['Year']

# Eliminar las columnas 'Year' y 'Car_Name'
train = train.drop(columns=['Year', 'Car_Name'])
test = test.drop(columns=['Year', 'Car_Name'])


# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
x_train=train.drop('Selling_Price',axis=1)
y_train = train['Selling_Price']

print(x_train)
print(y_train)
x_test=test.drop('Selling_Price',axis=1)
y_test = test['Selling_Price']

# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las variables numéricas al intervalo [0, 1].
# - Selecciona las K mejores entradas.
# - Ajusta un modelo de regresion lineal.

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Identificar las columnas numéricas y categóricas
numerical_features = ['Present_Price', 'Driven_kms', 'Owner']
categorical_features = ['Fuel_Type', 'Selling_type', 'Transmission']

# Crear transformadores para columnas numéricas y categóricas
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', MinMaxScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Crear el preprocesador que aplica las transformaciones apropiadas a cada tipo de columna
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Crear el pipeline completo
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('select_k_best', SelectKBest(f_regression)),
    ('regressor', LinearRegression())
])

# Ajustar el modelo al conjunto de entrenamiento
pipeline.fit(x_train, y_train)
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use el error medio absoluto
# para medir el desempeño modelo.
from sklearn.model_selection import GridSearchCV

# Definir el espacio de parámetros para la búsqueda
param_grid = {
    'select_k_best__k': [5, 6, 7],  # Número de características a seleccionar
    'regressor': [LinearRegression()]
}

# Configurar GridSearchCV con validación cruzada (10 splits)
grid_search = GridSearchCV(pipeline, param_grid, cv=10, scoring='neg_mean_absolute_error')

# Ajustar el modelo con validación cruzada
grid_search.fit(x_train, y_train)

# Ver el mejor parámetro encontrado
grid_search.best_params_

# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
import joblib
import gzip

# Guardar el modelo entrenado en un archivo comprimido
model_filename = 'files/models/model.pkl.gz'
with gzip.open(model_filename, 'wb') as f:
    joblib.dump(grid_search.best_estimator_, f)
# Paso 6.
# Calcule las metricas r2, error cuadratico medio, y error absoluto medio
# para los conjuntos de entrenamiento y prueba. Guardelas en el archivo
# files/output/metrics.json. Cada fila del archivo es un diccionario con
# las metricas de un modelo. Este diccionario tiene un campo para indicar
# si es el conjunto de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'r2': 0.8, 'mse': 0.7, 'mad': 0.9}
# {'type': 'metrics', 'dataset': 'test', 'r2': 0.7, 'mse': 0.6, 'mad': 0.8}
#
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json

# Realizar predicciones sobre los conjuntos de entrenamiento y prueba
y_train_pred = grid_search.predict(x_train)
y_test_pred = grid_search.predict(x_test)

# Calcular las métricas de desempeño
metrics = [
    {
        'type': 'metrics', 'dataset': 'train', 
        'r2': r2_score(y_train, y_train_pred),
        'mse': mean_squared_error(y_train, y_train_pred),
        'mad': mean_absolute_error(y_train, y_train_pred)
    },
    {
        'type': 'metrics', 'dataset': 'test', 
        'r2': r2_score(y_test, y_test_pred),
        'mse': mean_squared_error(y_test, y_test_pred),
        'mad': mean_absolute_error(y_test, y_test_pred)
    }
]

# Guardar las métricas en un archivo JSON
metrics_filename = 'files/output/metrics.json'
with open(metrics_filename, 'w') as f:
    json.dump(metrics, f, indent=4)