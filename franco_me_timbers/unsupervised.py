#importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.cluster import Birch
from sklearn.cluster import AgglomerativeClustering

#importing the data from the .csv file
df = pd.read_csv('videogames.csv')

# Convert User Score to double and normalize it to match Critic Score
df['User_Score'] = pd.to_numeric(df['User_Score'], errors='coerce')*10

#dropping the rows with NaN values
df = df.dropna(axis=0, how='any')

#separating the important parts and dropping other variables
features = ['Year_of_Release', 'NA_Sales',
       'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales', 'Critic_Score',
       'Critic_Count', 'User_Score', 'User_Count']

classes = df['Platform']
dropped = df[['Year_of_Release', 'NA_Sales',
       'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales', 'Critic_Score',
       'Critic_Count', 'User_Score', 'User_Count']]

kmeans = KMeans(n_clusters=17, random_state=None, init='k-means++').fit(dropped)
birch = Birch(threshold=3 ,n_clusters=17).fit(dropped)
agglo = AgglomerativeClustering(n_clusters=17).fit(dropped)

print('----->KMeans results')
print('ARI: '+str(metrics.adjusted_rand_score(classes, kmeans.labels_)))
print('AMI: '+str(metrics.adjusted_mutual_info_score(classes, kmeans.labels_)))
print('Homogeneity: '+str(metrics.homogeneity_score(classes, kmeans.labels_)))
print('Completeness: '+str(metrics.completeness_score(classes, kmeans.labels_)))

print('----->Birch results')
print('ARI: '+str(metrics.adjusted_rand_score(classes, birch.labels_)))
print('AMI: '+str(metrics.adjusted_mutual_info_score(classes, birch.labels_)))
print('Homogeneity: '+str(metrics.homogeneity_score(classes, birch.labels_)))
print('Completeness: '+str(metrics.completeness_score(classes, birch.labels_)))

print('----->Agglomerative Clustering results')
print('ARI: '+str(metrics.adjusted_rand_score(classes, agglo.labels_)))
print('AMI: '+str(metrics.adjusted_mutual_info_score(classes, agglo.labels_)))
print('Homogeneity: '+str(metrics.homogeneity_score(classes, agglo.labels_)))
print('Completeness: '+str(metrics.completeness_score(classes, agglo.labels_)))