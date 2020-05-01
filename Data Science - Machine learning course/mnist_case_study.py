import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

train = pd.read_csv('mnist_train.csv').sample(n=10000)
test = pd.read_csv('mnist_test.csv').sample(n=1000)

print(train.describe())
#there are several features with 0 std (no variability)

#pre-processing
selector = VarianceThreshold(threshold = 0.3)
features = selector.fit(train.drop(columns=['label']))
select_cols = train.drop(columns=['label']).columns.values[features.get_support()] #get_support returns a boolean vector indicating which features (columns' names) were selected
X_train = pd.DataFrame(features.transform(train.drop(columns=['label'])),columns = select_cols) #build dataframe with selected features only
X_test = pd.DataFrame(features.transform(test.drop(columns=['label'])),columns = select_cols) #build dataframe with selected features only


print(X_train.describe()) #more than 100 features are removed due to low variability

#fit models
#1 - KNN
params = {"n_neighbors": [3]} #previous run identified 3 as the optimum k #np.arange(3, 10, 2)

m = KNeighborsClassifier()
bm = GridSearchCV(m, params, cv=5)
bm.fit(X_train, train['label'])
acc = bm.score(X_test, test['label'])
y_pred = bm.predict(X_test)
print(bm.best_params_)
print('KNN Accuracy = %f' %acc)
print(confusion_matrix(test['label'], y_pred, labels=test['label'].unique()))

#2 - Random Forests
params = {'n_estimators':[1000], "min_samples_split": [3]}, "max_features": [25]} #previous run identified 3 splits and 25 features as optimal
m = RandomForestClassifier()
bm = GridSearchCV(m, params, cv=5)
bm.fit(X_train, train['label'])
acc = bm.score(X_test, test['label'])
y_pred = bm.predict(X_test)
print(bm.best_params_)
print('Random Forest Accuracy = %f' %acc)
print(confusion_matrix(test['label'], y_pred, labels=test['label'].unique()))
print(X_train.columns)
print(bm.best_estimator_.feature_importances_)

#the random forest algorithm is a small improvement on the KNN one