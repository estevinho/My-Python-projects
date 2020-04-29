import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('OliveOils.csv')

#build region variable using area, since this is not in the data set
def region(r):
	if r in ['Coastal-Sardinia','Inland-Sardinia']:
		v='Sardinia'
	elif r in ['Calabria', 'Sicily','North-Apulia','South-Apulia']:
		v='Southern Italy'
	else:
		v='Northern Italy'
	return v
	
data['region']=data['Area name'].apply(region)

#descriptive statistics
print(data.groupby('region').describe())


fig, axs = plt.subplots(1,7, figsize=(10,5), sharey=False)
data.boxplot(by = 'region',ax = axs)
fig.autofmt_xdate() #automatically adjust axes direction
plt.show()
#scatter plots for linoleic and eicosenoic (understanding class separability) #try linoleic and palmitic

sns.scatterplot(x="palmitic", y="linoleic", hue="region", data=data)
plt.show()

train, test = train_test_split(data.drop(columns=['Area name']), test_size=0.2)

#fit k-nearest neighbours tunning k for the best result
params = {"n_neighbors": np.arange(1, 31, 2)}

m = KNeighborsClassifier()
bm = GridSearchCV(m, params)
bm.fit(train.drop(columns=['region']), train['region'])
acc = bm.score(test.drop(columns=['region']), test['region'])

print(bm.best_params_)
print('KNN Accuracy = %f' %acc)

#fit decision tree tunning 
params = {"max_depth": np.arange(2, 11, 1), "min_samples_split": np.arange(10, 50, 5), "min_samples_leaf": np.arange(10, 50, 5),'max_leaf_nodes': np.arange(3, 21, 2)}

m = DecisionTreeClassifier()
bm = GridSearchCV(m, params)
bm.fit(train.drop(columns=['region']), train['region'])
acc = bm.score(test.drop(columns=['region']), test['region'])

print(bm.best_params_)
print('Decision Tree Accuracy = %f' %acc)
plot_tree(bm.best_estimator_, filled=True, feature_names = test.drop(columns=['region']).columns)
plt.show()

#fit random forest tunning number of trees
params = {'n_estimators':np.arange(20, 101, 10) ,"max_depth": [4], "min_samples_split": [10], "min_samples_leaf": [10], 'max_leaf_nodes': [5]} #from previous step

m = RandomForestClassifier()
bm = GridSearchCV(m, params)
bm.fit(train.drop(columns=['region']), train['region'])
acc = bm.score(test.drop(columns=['region']), test['region'])

print(bm.best_params_)
print('Random Forest Accuracy = %f' %acc)

#random forest performance doesn't improve much compared to single decision tree

