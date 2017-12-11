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
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from tools import gen_log_space, cross_validate_model
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

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

#kmeans = KMeans(n_clusters=17, random_state=None, init='k-means++').fit(dropped)
#birch = Birch(threshold=3 ,n_clusters=17).fit(dropped)
agglo = AgglomerativeClustering(n_clusters=17).fit(dropped)

classes = agglo.labels_

x_train, x_test, y_train, y_test = train_test_split(dropped, classes, test_size=0.2, random_state=69)

#DimensionalityReduction = PCA()
#DimensionalityReduction = LinearDiscriminantAnalysis()
#DimensionalityReduction.fit(x_train,y_train)


#x_train = DimensionalityReduction.transform(x_train)
#x_test = DimensionalityReduction.transform(x_test)

model_k = KNeighborsClassifier(n_neighbors=20)
model_k.fit(x_train, y_train)

print('------------->KNN')
print('accuracy {:.2f}%'.format(metrics.accuracy_score(y_test, model_k.predict(x_test))*100))
print('train accuracy {:.2f}%'.format(metrics.accuracy_score(y_train, model_k.predict(x_train))*100))

model_nb = GaussianNB()
model_nb.fit(x_train, y_train)

print('-------------->NB')
print('accuracy {:.2f}%'.format(metrics.accuracy_score(y_test, model_nb.predict(x_test))*100))
print('train accuracy {:.2f}%'.format(metrics.accuracy_score(y_train, model_nb.predict(x_train))*100))


model_dt = DecisionTreeClassifier(max_depth=11)
model_dt.fit(x_train, y_train)

print('--------------->DT')
print('accuracy {:.2f}%'.format(metrics.accuracy_score(y_test, model_dt.predict(x_test))*100))
print('train accuracy {:.2f}%'.format(metrics.accuracy_score(y_train, model_dt.predict(x_train))*100))


model_rf =  RandomForestClassifier(max_depth=12)
model_rf.fit(x_train, y_train)

print('------------->RF')
print('accuracy {:.2f}%'.format(metrics.accuracy_score(y_test, model_rf.predict(x_test))*100))
print('train accuracy {:.2f}%'.format(metrics.accuracy_score(y_train, model_rf.predict(x_train))*100))


model_svm = SVC(decision_function_shape='ovr', kernel='linear')
model_svm.fit(x_train, y_train)

print('-------------->SVM (linear)')
print('accuracy {:.2f}%'.format(metrics.accuracy_score(y_test, model_svm.predict(x_test))*100))
print('train accuracy {:.2f}%'.format(metrics.accuracy_score(y_train, model_svm.predict(x_train))*100))

model_rbf = SVC(decision_function_shape='ovr', kernel='rbf', gamma=0.001, C=10)
model_rbf.fit(x_train, y_train)

print('--------------->SVM (RBF)')
print('accuracy {:.2f}%'.format(metrics.accuracy_score(y_test, model_rbf.predict(x_test))*100))
print('train accuracy {:.2f}%'.format(metrics.accuracy_score(y_train, model_rbf.predict(x_train))*100))
