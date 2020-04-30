import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('titanic.csv')

#descriptive statistics
print(data.describe(include='all'))

#create train test split stratifying based on survived
train, test = train_test_split(data, test_size=0.2)

print(train.groupby(['Sex','Pclass']).describe(include='all'))
print(test.describe(include='all'))

#algorithm 1 - random guess
y_hat1 = data['Survived'].sample(frac=0.2).values.tolist()
		
#evaluate accuracy of random guess
y_test = test['Survived'].values.tolist()

cc = 0 #number of correct classifications
for i in range(len(y_test)-1):
	if y_hat1[i] == y_test[i]:
		cc+=1

print('Overall accuracy of random guess:')
print(cc/len(y_test))

confusion_matrix = {'P = 1, A = 1':0,'P = 1, A = 0':0,'P = 0, A = 1':0,'P = 0, A = 0':0}
for i in range(len(y_test)-1):
	if y_hat1[i]==1 and y_test[i]==1:
		confusion_matrix['P = 1, A = 1']+=1
	elif y_hat1[i]==1 and y_test[i]==0:
		confusion_matrix['P = 1, A = 0']+=1
	elif y_hat1[i]==0 and y_test[i]==1:
		confusion_matrix['P = 0, A = 1']+=1
	else:
		confusion_matrix['P = 0, A = 0']+=1

recall = confusion_matrix['P = 1, A = 1']/(confusion_matrix['P = 1, A = 1']+confusion_matrix['P = 0, A = 1'])
precision = confusion_matrix['P = 1, A = 1']/(confusion_matrix['P = 1, A = 1']+confusion_matrix['P = 1, A = 0'])

print('Sensitivity = %f' %recall)
print('Specificity = %f' %precision)
print('F1 score = %f' %(2*(precision*recall)/(recall+precision)))

#algorithm 2 - predicting survival for females (1) and death for males (0), since in the train data females are much more likely to survive
y_hat2 = [] 
for i in test['Sex']:
	if i == 'female':
		y_hat2.append(1)
	else:
		y_hat2.append(0)

#evaluate accuracy of gender based prediction
cc = 0 #number of correct classifications
for i in range(len(y_test)-1):
	if y_hat2[i] == y_test[i]:
		cc+=1

print('Overall accuracy of gender based prediction:')
print(cc/len(y_test))

confusion_matrix = {'P = 1, A = 1':0,'P = 1, A = 0':0,'P = 0, A = 1':0,'P = 0, A = 0':0}
for i in range(len(y_test)-1):
	if y_hat2[i]==1 and y_test[i]==1:
		confusion_matrix['P = 1, A = 1']+=1
	elif y_hat2[i]==1 and y_test[i]==0:
		confusion_matrix['P = 1, A = 0']+=1
	elif y_hat2[i]==0 and y_test[i]==1:
		confusion_matrix['P = 0, A = 1']+=1
	else:
		confusion_matrix['P = 0, A = 0']+=1

recall = confusion_matrix['P = 1, A = 1']/(confusion_matrix['P = 1, A = 1']+confusion_matrix['P = 0, A = 1'])
precision = confusion_matrix['P = 1, A = 1']/(confusion_matrix['P = 1, A = 1']+confusion_matrix['P = 1, A = 0'])

print('Sensitivity = %f' %recall)
print('Specificity = %f' %precision)
print('F1 score = %f' %(2*(precision*recall)/(recall+precision)))

#algorithm 3 - predicting survival for class 1 (1) and death for others (0), since in the train data class 1 is the only with survival rate above 50%
y_hat3 = []
for i in test['Pclass']:
	if i == 1:
		y_hat3.append(1)
	else:
		y_hat3.append(0)

#evaluate accuracy of class based prediction
cc = 0 #number of correct classifications
for i in range(len(y_test)-1):
	if y_hat3[i] == y_test[i]:
		cc+=1

print('Overall accuracy of class based prediction:')
print(cc/len(y_test))

confusion_matrix = {'P = 1, A = 1':0,'P = 1, A = 0':0,'P = 0, A = 1':0,'P = 0, A = 0':0}
for i in range(len(y_test)-1):
	if y_hat3[i]==1 and y_test[i]==1:
		confusion_matrix['P = 1, A = 1']+=1
	elif y_hat3[i]==1 and y_test[i]==0:
		confusion_matrix['P = 1, A = 0']+=1
	elif y_hat3[i]==0 and y_test[i]==1:
		confusion_matrix['P = 0, A = 1']+=1
	else:
		confusion_matrix['P = 0, A = 0']+=1

recall = confusion_matrix['P = 1, A = 1']/(confusion_matrix['P = 1, A = 1']+confusion_matrix['P = 0, A = 1'])
precision = confusion_matrix['P = 1, A = 1']/(confusion_matrix['P = 1, A = 1']+confusion_matrix['P = 1, A = 0'])

print('Sensitivity = %f' %recall)
print('Specificity = %f' %precision)
print('F1 score = %f' %(2*(precision*recall)/(recall+precision)))

#algorithm 4 - predicting survival for females in class 1 and 2 (1) and death for others (0), since these are the groups with survival rate above 50%
y_hat4 = []
sex = test['Sex'].values.tolist()
pclass = test['Pclass'].values.tolist()
for i in range(len(y_test)-1):
	if sex[i] == 'female' and pclass[i] in [1,2]:
		y_hat4.append(1)
	else:
		y_hat4.append(0)

#evaluate accuracy of gender based prediction
cc = 0 #number of correct classifications
for i in range(len(y_test)-1):
	if y_hat4[i] == y_test[i]:
		cc+=1

print('Overall accuracy of gender and class based prediction:')
print(cc/len(y_test))

confusion_matrix = {'P = 1, A = 1':0,'P = 1, A = 0':0,'P = 0, A = 1':0,'P = 0, A = 0':0}
for i in range(len(y_test)-1):
	if y_hat4[i]==1 and y_test[i]==1:
		confusion_matrix['P = 1, A = 1']+=1
	elif y_hat4[i]==1 and y_test[i]==0:
		confusion_matrix['P = 1, A = 0']+=1
	elif y_hat4[i]==0 and y_test[i]==1:
		confusion_matrix['P = 0, A = 1']+=1
	else:
		confusion_matrix['P = 0, A = 0']+=1

recall = confusion_matrix['P = 1, A = 1']/(confusion_matrix['P = 1, A = 1']+confusion_matrix['P = 0, A = 1'])
precision = confusion_matrix['P = 1, A = 1']/(confusion_matrix['P = 1, A = 1']+confusion_matrix['P = 1, A = 0'])

print('Sensitivity = %f' %recall)
print('Specificity = %f' %precision)
print('F1 score = %f' %(2*(precision*recall)/(recall+precision)))

#algorithm 5 - Linear Discriminant Analysis on Fare
clf = LinearDiscriminantAnalysis()
clf.fit(train['Fare'].values.reshape(-1,1), train['Survived'])

print('Overall accuracy of LDA using fare:')
print(clf.score(test['Fare'].values.reshape(-1,1), test['Survived']))

#algorithm 6 - Quadratic Discriminant Analysis on Fare
clf = QuadraticDiscriminantAnalysis()
clf.fit(train['Fare'].values.reshape(-1,1), train['Survived'])

print('Overall accuracy of QDA using fare:')
print(clf.score(test['Fare'].values.reshape(-1,1), test['Survived']))

#algorithm 7 - Logistic Regression on age
clf = LogisticRegression()
clf.fit(train['Age'].values.reshape(-1,1), train['Survived'])

print('Overall accuracy of log reg using age:')
print(clf.score(test['Age'].values.reshape(-1,1), test['Survived']))

#algorithm 8 - Logistic Regression on sex, class, fare and age
#build dummies for categorical variables
def create_dummies(df,column_name):
	"""Create Dummy Columns (One Hot Encoding) from a single Column

	Usage
	------
	train = create_dummies(train,"Age")
	"""
	dummies = pd.get_dummies(df[column_name],prefix=column_name)
	df = pd.concat([df,dummies],axis=1)
	return df

toDummy = ['Sex','Pclass']
for column in toDummy:
	train = create_dummies(train, column)
	test = create_dummies(test, column)
	
clf = LogisticRegression()
clf.fit(train[['Sex_female','Pclass_1','Pclass_2','Fare','Age']], train['Survived'])

print('Overall accuracy of log reg:')
print(clf.score(test[['Sex_female','Pclass_1','Pclass_2','Fare','Age']], test['Survived']))

#algorithm 9 - Logistic Regression with all variables
clf = LogisticRegression()
clf.fit(train.drop(columns=['Survived','Pclass_3','Sex_male','Sex','Pclass','Name']), train['Survived'])

print('Overall accuracy of log reg:')
print(clf.score(test.drop(columns=['Survived','Pclass_3','Sex_male','Sex','Pclass','Name']), test['Survived']))

#algorithm 10 - KNN with all variables
params = {"n_neighbors": np.arange(3, 51, 2)}

m = KNeighborsClassifier()
bm = GridSearchCV(m, params, cv=10)
bm.fit(train.drop(columns=['Survived','Pclass_3','Sex_male','Sex','Pclass','Name']), train['Survived'])
acc = bm.score(test.drop(columns=['Survived','Pclass_3','Sex_male','Sex','Pclass','Name']), test['Survived'])

print(bm.best_params_)
print('KNN Accuracy = %f' %acc)

#algorithm 11 - tree with all variables
params = {"ccp_alpha": np.arange(0, 0.05, 0.002)}

m = DecisionTreeClassifier()
bm = GridSearchCV(m, params)
bm.fit(train.drop(columns=['Survived','Pclass_3','Sex_male','Sex','Pclass','Name']), train['Survived'])
acc = bm.score(test.drop(columns=['Survived','Pclass_3','Sex_male','Sex','Pclass','Name']), test['Survived'])

print(bm.best_params_)
print('Decision tree Accuracy = %f' %acc)

plt.figure(figsize=(10,8))
plot_tree(bm.best_estimator_, filled=True, feature_names = test.drop(columns=['Survived','Pclass_3','Sex_male','Sex','Pclass','Name']).columns)
plt.show()

#algorithm 12 - random forest
params = {'n_estimators':[100] ,"max_features": [1,2,3,4,5,6,7] }

m = RandomForestClassifier()
bm = GridSearchCV(m, params)
bm.fit(train.drop(columns=['Survived','Pclass_3','Sex_male','Sex','Pclass','Name']), train['Survived'])
acc = bm.score(test.drop(columns=['Survived','Pclass_3','Sex_male','Sex','Pclass','Name']), test['Survived'])

print(bm.best_params_)
print('Random Forest Accuracy = %f' %acc)
print(train.drop(columns=['Survived','Pclass_3','Sex_male','Sex','Pclass','Name']).columns)
print(bm.best_estimator_.feature_importances_)