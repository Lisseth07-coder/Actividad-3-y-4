# Importar las librerías necesarias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Crear un dataset ficticio
data = {
    'origen': ['A', 'A', 'B', 'B', 'C', 'C', 'D', 'D'],
    'destino': ['B', 'C', 'A', 'D', 'A', 'D', 'B', 'C'],
    'costo': [5, 10, 5, 3, 10, 1, 3, 1],
    'distancia': [2, 5, 2, 1, 5, 1, 2, 1],
    'pasajeros': [100, 150, 200, 50, 300, 75, 100, 60]
}

# Convertir el dataset en un DataFrame de pandas
df = pd.DataFrame(data)

# Definir las variables independientes (X) y la variable dependiente (y)
X = df[['costo', 'distancia']]  # Variables predictoras
y = df['pasajeros']  # Variable objetivo

# Dividir el conjunto de datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear un modelo de regresión lineal
modelo = LinearRegression()

# Entrenar el modelo con el conjunto de entrenamiento
modelo.fit(X_train, y_train)

# Predecir los valores en el conjunto de prueba
y_pred = modelo.predict(X_test)

# Evaluar el modelo utilizando el error cuadrático medio (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"Error cuadrático medio: {mse}")

