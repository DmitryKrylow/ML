import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.io import arff
import pandas as pd

# === 1. Загрузка данных из ARFF ===
data, meta = arff.loadarff("messidor_features.arff")
df = pd.DataFrame(data)

# Признаки X и классы y (если нужны)
X = df.drop("Class", axis=1).values
y = df["Class"].astype(int).values  # можно использовать для проверки качества

# === 2. Масштабирование признаков ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# === 3. Алгоритмы кластеризации ===

# KMeans
kmeans = KMeans(n_clusters=2, random_state=42)
labels_kmeans = kmeans.fit_predict(X_scaled)

# Agglomerative Clustering
agg = AgglomerativeClustering(n_clusters=2)
labels_agg = agg.fit_predict(X_scaled)

# DBSCAN
dbscan = DBSCAN(eps=1.5, min_samples=5)  # параметры можно подбирать
labels_dbscan = dbscan.fit_predict(X_scaled)

# === 4. Оценка качества ===
def evaluate(name, labels):
    if len(set(labels)) > 1 and -1 not in set(labels):  # если больше 1 кластера и DBSCAN не дал "шум"
        print(f"\nМетрики для {name}:")
        print("Silhouette:", silhouette_score(X_scaled, labels))
        print("Davies-Bouldin:", davies_bouldin_score(X_scaled, labels))
        print("Calinski-Harabasz:", calinski_harabasz_score(X_scaled, labels))
    else:
        print(f"\n{name}: кластеризация неустойчивая (все в 1 кластер или шум)")

evaluate("KMeans", labels_kmeans)
evaluate("Agglomerative", labels_agg)
evaluate("DBSCAN", labels_dbscan)

# === 5. Визуализация (через PCA в 2D) ===
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.scatter(X_pca[:,0], X_pca[:,1], c=labels_kmeans)
plt.title("KMeans")

plt.subplot(1,3,2)
plt.scatter(X_pca[:,0], X_pca[:,1], c=labels_agg)
plt.title("Agglomerative")

plt.subplot(1,3,3)
plt.scatter(X_pca[:,0], X_pca[:,1], c=labels_dbscan)
plt.title("DBSCAN")

plt.show()