
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn import preprocessing

data = pd.read_csv("airlines.csv")


missing_values = data.isnull().sum()
print("Missing Values:\n", missing_values)


data.hist(figsize=(15, 10))
plt.tight_layout()
plt.show()


award_counts = data['award'].value_counts(normalize=True) * 100
print("Percentage of Customers with/without Award:\n", award_counts)


correlation = data.corr()['balance'].sort_values(ascending=False)
print("Correlation with balance feature:\n", correlation)


plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()


plt.figure(figsize=(10, 6))
plt.scatter(data['bonus_miles'], data['bonus_trans'])
plt.xlabel('Frequent Flying Bonus Miles')
plt.ylabel('Non-flight Bonus Transactions')
plt.title('Relationship between Frequent Flying Bonuses and Non-flight Bonus Transactions')
plt.show()


scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data.drop(columns=['id', 'award']))


inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.show()


silhouette_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data_scaled)
    score = silhouette_score(data_scaled, kmeans.labels_)
    silhouette_scores.append(score)

plt.figure(figsize=(10, 6))
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Optimal k')
plt.show()
