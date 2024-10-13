import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

# Crear un dataset ficticio para transporte masivo
data = {
    'parada': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
    'intervalo_buses': [10, 15, 12, 20, 5, 10, 7, 8], # minutos
    'capacidad': [50, 50, 40, 60, 45, 55, 60, 50],    # número de pasajeros
    'pasajeros_abordados': [30, 25, 20, 35, 15, 10, 40, 45], # pasajeros abordados
    'distancia': [1.5, 2.0, 2.5, 3.0, 1.2, 0.8, 1.8, 2.2]  # distancia en km entre paradas
}

# Convertir a DataFrame
df = pd.DataFrame(data)

# Seleccionar las columnas para el modelo de clustering
X = df[['intervalo_buses', 'capacidad', 'pasajeros_abordados', 'distancia']]

# Aplicar el modelo K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(X)

# Mostrar el dataset con los clusters
print(df)

# Visualizar los clusters (opcional)
plt.scatter(df['pasajeros_abordados'], df['intervalo_buses'], c=df['cluster'], cmap='viridis')
plt.xlabel('Pasajeros abordados')
plt.ylabel('Intervalo entre buses')
plt.title('Clusters de paradas de buses')
plt.show()
