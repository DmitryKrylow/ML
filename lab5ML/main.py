import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score, \
    jaccard_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# 1. Загрузка и объединение CSV
data_folder = "C:\\Users\\dima2\\PycharmProjects\\lab5ML\\raw"
all_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]

df_list = []
for file in all_files:
    df = pd.read_csv(os.path.join(data_folder, file))
    df.columns = df.columns.str.strip() 
    df['phase'] = df['phase'].astype(str).str.strip()
    df_list.append(df)

data = pd.concat(df_list, ignore_index=True)
X = data.drop(columns=['phase', 'timestamp']).values

data['phase'] = data['phase'].str.strip()
le = LabelEncoder()
y = le.fit_transform(data['phase'])

print("Данные загружены. Размер:", X.shape)

#2. Масштабирование признаков
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# -----------------------------
# 3. Подбор параметров DBSCAN
# -----------------------------
eps_values = np.arange(0.2, 3.0, 0.1)
min_samples_values = [3, 5, 10, 15]

best_score = -1
best_params_dbscan = None
best_labels_dbscan = None

for eps in eps_values:
    for min_samples in min_samples_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X_scaled)
        # Проверяем, что кластеров больше 1 и нет всех шумов
        if len(set(labels)) > 1 and -1 not in set(labels):
            score = silhouette_score(X_scaled, labels)
            if score > best_score:
                best_score = score
                best_params_dbscan = (eps, min_samples)
                best_labels_dbscan = labels

print(f"\nЛучшие параметры DBSCAN: eps={best_params_dbscan[0]}, min_samples={best_params_dbscan[1]}")
print(f"Лучший Silhouette Score: {best_score:.3f}")

# 3. Алгоритмы кластеризации
# KMeans

scores = []
for n in range(2, 10):
    kmeans = KMeans(n_clusters=n, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    scores.append((n, score))
    print(f"n_clusters={n}: Silhouette={score:.3f}")

best_n = max(scores, key=lambda x: x[1])
print("\nЛучшее число кластеров:", best_n[0])
labels_kmeans = KMeans(n_clusters=best_n[0], random_state=42).fit_predict(X_scaled)

# Agglomerative Clustering
scores = []
for n in range(2, 10):  # перебираем количество кластеров
    agg = AgglomerativeClustering(n_clusters=n)
    labels = agg.fit_predict(X_scaled)

    # вычисляем Silhouette score
    score = silhouette_score(X_scaled, labels)
    scores.append((n, score))
    print(f"n_clusters={n}: Silhouette={score:.3f}")
best_n = max(scores, key=lambda x: x[1])
print("\nЛучшее число кластеров:", best_n[0])
labels_agg = AgglomerativeClustering(n_clusters=best_n[0]).fit_predict(X_scaled)

#4. Оценка качества
def evaluate(name, labels):
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    if n_clusters > 1:
        print(f"\nМетрики для {name}:")
        print("Silhouette:", silhouette_score(X_scaled, labels))
        print("Davies-Bouldin:", davies_bouldin_score(X_scaled, labels))
        print("Calinski-Harabasz:", calinski_harabasz_score(X_scaled, labels))
    else:
        print(f"\n{name}: кластеризация неустойчивая (все в 1 кластер или шум)")

evaluate("KMeans", labels_kmeans)
evaluate("Agglomerative", labels_agg)
evaluate("DBSCAN", best_labels_dbscan)

# 5. Визуализация (через PCA в 2D)
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
plt.scatter(X_pca[:,0], X_pca[:,1], c=best_labels_dbscan)
plt.title("DBSCAN")

plt.show()


