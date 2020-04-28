import pandas as pd
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict
from scipy.stats import norm


data = pd.read_csv('weight-height.csv')

#descriptive statistics
print(data.groupby('Gender').describe())

train, test = train_test_split(data, test_size=0.5) #partition the dataset in 2 to generate train and test sets

y = train['Gender']
x = train['Height']

#develop algorithm using train set
#algorithm 1 - random guess
y_hat1 = data['Gender'].sample(frac=0.5).values.tolist()
#create dummy list
y_hat1_dummy = []
for i in y_hat1:
	if i == 'Male':
		y_hat1_dummy.append(1)
	else:
		y_hat1_dummy.append(0)
		
#evaluate accuracy of random guess
y_test = test['Gender'].values.tolist()
#create dummy list
y_test_dummy = []
for i in y_test:
	if i == 'Male':
		y_test_dummy.append(1)
	else:
		y_test_dummy.append(0)

cc = 0 #number of correct classifications
for i in range(5000):
	if y_hat1_dummy[i] == y_test_dummy[i]:
		cc+=1

print('Overall accuracy of random guess:')
print(cc/5000)

#algorithm 2 - classification using male descriptive statistics
#descriptive statistics show that males are on average higher than females, so height should allow better classifications than a random guess
#predict male if height is within 2 standard deviations of the average for males

male_mean = data.groupby('Gender')['Height'].mean()['Male']
male_std = data.groupby('Gender')['Height'].std()['Male']

y_test_height = test['Height'].values.tolist()
y_hat2_dummy=[]
for i in y_test_height:
	if i >= male_mean-2*male_std and i<=male_mean+2*male_std:
		y_hat2_dummy.append(1)
	else:
		y_hat2_dummy.append(0)
		
cc2 = 0 #number of correct classifications
for i in range(5000):
	if y_hat2_dummy[i] == y_test_dummy[i]:
		cc2+=1

print('Overall accuracy of algorithm 2:')
print(cc2/5000)

#evaluate different versions of algorithm 2 using the train set
y_train_height = train['Height'].values.tolist()
y_train = train['Gender'].values.tolist()
#create dummy list
y_train_dummy = []
for i in y_train:
	if i == 'Male':
		y_train_dummy.append(1)
	else:
		y_train_dummy.append(0)
		
y_hat_dummy={}
cc={}

for j in [0.5,1,1.5,2,2.5,3]:
	y_hat_dummy[j] = []
	cc[j] = 0
	for i in y_train_height:
		if i >= male_mean-j*male_std and i<=male_mean+j*male_std:
			y_hat_dummy[j].append(1)
		else:
			y_hat_dummy[j].append(0)	
	for i in range(5000):
		if y_hat_dummy[j][i] == y_train_dummy[i]:
			cc[j]+=1
	print('Overall accuracy of algorithm 2 with %s standard deviations:' %j)
	print(cc[j]/5000)

#plot of number of correct classifications

lists = sorted(cc.items()) # sorted by key, return a list of tuples
x, y = zip(*lists) # unpack a list of pairs into two tuples

plt.plot(x, y)
plt.show()

#using the training set the most accurate predictor is the algorithm that uses 1.5 standard deviations

#check accuracy of this predictor in the test set
y_hat_best_dummy=[]
for i in y_test_height:
	if i >= male_mean-1.5*male_std and i<=male_mean+1.5*male_std:
		y_hat_best_dummy.append(1)
	else:
		y_hat_best_dummy.append(0)
		
cc_best = 0 #number of correct classifications
for i in range(5000):
	if y_hat_best_dummy[i] == y_test_dummy[i]:
		cc_best+=1

print('Overall accuracy of best algorithm in test set:')
print(cc_best/5000)

#similar accuracy in training and test set

#build a confusion matrix for the best algorithm
confusion_matrix = {'P = Male, A = Male':0,'P = Male, A = Female':0,'P = Female, A = Male':0,'P = Female, A = Female':0}
for i in range(5000):
	if y_hat_best_dummy[i]==1 and y_test_dummy[i]==1:
		confusion_matrix['P = Male, A = Male']+=1
	elif y_hat_best_dummy[i]==1 and y_test_dummy[i]==0:
		confusion_matrix['P = Male, A = Female']+=1
	elif y_hat_best_dummy[i]==0 and y_test_dummy[i]==1:
		confusion_matrix['P = Female, A = Male']+=1
	else:
		confusion_matrix['P = Female, A = Female']+=1

print(confusion_matrix)		

#accuracy per sex
print('Female accuracy: %f' %(confusion_matrix['P = Female, A = Female']/(confusion_matrix['P = Female, A = Female']+confusion_matrix['P = Male, A = Female'])))
print('Male accuracy: %f' %(confusion_matrix['P = Male, A = Male']/(confusion_matrix['P = Male, A = Male']+confusion_matrix['P = Female, A = Male'])))

#as expected due to the nature of the algorithm, the accuracy for male is much higher than for female

#calculate TPR, TNR and PPV
print('TPR = %f' %(confusion_matrix['P = Male, A = Male']/(confusion_matrix['P = Male, A = Male']+confusion_matrix['P = Female, A = Male'])))
print('TNR = %f' %(confusion_matrix['P = Female, A = Female']/(confusion_matrix['P = Female, A = Female']+confusion_matrix['P = Male, A = Female'])))
print('PPV = %f' %(confusion_matrix['P = Male, A = Male']/(confusion_matrix['P = Male, A = Male']+confusion_matrix['P = Male, A = Female'])))

#calculate F1 score
recall = confusion_matrix['P = Male, A = Male']/(confusion_matrix['P = Male, A = Male']+confusion_matrix['P = Female, A = Male'])
precision = confusion_matrix['P = Male, A = Male']/(confusion_matrix['P = Male, A = Male']+confusion_matrix['P = Male, A = Female'])

print('F1 score = %f' %(2*(precision*recall)/(recall+precision)))

#training algorithm 2 again but based on the F-score

all_confusion_matrix = {}
all_recall={}
all_precision={}
all_f1_score ={}
all_FPR={}

for j in [0.5,1,1.5,2,2.5,3]:
	all_confusion_matrix[j] = {'P = Male, A = Male':0,'P = Male, A = Female':0,'P = Female, A = Male':0,'P = Female, A = Female':0}
	for i in range(5000):
		if y_hat_dummy[j][i]==1 and y_train_dummy[i]==1:
			all_confusion_matrix[j]['P = Male, A = Male']+=1
		elif y_hat_dummy[j][i]==1 and y_train_dummy[i]==0:
			all_confusion_matrix[j]['P = Male, A = Female']+=1
		elif y_hat_dummy[j][i]==0 and y_train_dummy[i]==1:
			all_confusion_matrix[j]['P = Female, A = Male']+=1
		else:
			all_confusion_matrix[j]['P = Female, A = Female']+=1
	all_recall[j] = all_confusion_matrix[j]['P = Male, A = Male']/(all_confusion_matrix[j]['P = Male, A = Male']+all_confusion_matrix[j]['P = Female, A = Male'])
	all_precision[j] = all_confusion_matrix[j]['P = Male, A = Male']/(all_confusion_matrix[j]['P = Male, A = Male']+all_confusion_matrix[j]['P = Male, A = Female'])
	all_f1_score[j] = 2*(all_precision[j]*all_recall[j])/(all_recall[j]+all_precision[j])
	all_FPR[j] = all_confusion_matrix[j]['P = Male, A = Female']/(all_confusion_matrix[j]['P = Male, A = Female']+all_confusion_matrix[j]['P = Female, A = Female'])
	
#plot f1_scores

f1_scores = sorted(all_f1_score.items()) # sorted by key, return a list of tuples
x2, y2 = zip(*f1_scores) # unpack a list of pairs into two tuples

plt.plot(x2, y2)
plt.xlabel('algorithm ID')
plt.ylabel('F1-score')
plt.title('F1-scores')
plt.show()

#the best algorithm is still the one that uses 1.5 standard deviations. This was expected since the dataset is balanced

#plot the ROC curve for the previous algorithms
x3 = all_FPR.values()
y3 = all_recall.values()

plt.plot(x3, y3)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC curve')
for j in [0.5,1,1.5,2,2.5,3]:
	plt.text(all_FPR[j],all_recall[j],j)
plt.show()

#the ROC curve shows that this classifier is better than the random classifier (straight line with slope 1)

#precision-recall plot
x4 = all_recall.values()
y4 = all_precision.values()

plt.plot(x4, y4)
plt.xlabel('recall')
plt.ylabel('precision')
plt.title('precision-recall plot')
for j in [0.5,1,1.5,2,2.5,3]:
	plt.text(all_recall[j],all_precision[j],j)
plt.show()

#section 3 - linear regression for categorical data
#transform gender in a dummy variable
def create_dummies(df,column_name):
	"""Create Dummy Columns (One Hot Encoding) from a single Column

	Usage
	------
	train = create_dummies(train,"category")
	"""
	dummies = pd.get_dummies(df[column_name],prefix=column_name)
	df = pd.concat([df,dummies],axis=1)
	return df
	
train = create_dummies(train,'Gender') #use 'Gender_female' to keep consistency with course
test = create_dummies(test,'Gender')

#probability of being female given that 66 inches tall
train_66 = train[round(train['Height'])==66]
#print(train_66)
print(train_66['Gender_Female'].mean())

#study previous for several values
female_prob = {}
for h in range(60,76):
	female_prob[h] = train[round(train['Height'])== h]['Gender_Female'].mean()

print(female_prob)
#plot probability of being female for different height values
listf = sorted(female_prob.items()) # sorted by key, return a list of tuples
xf, yf = zip(*listf) # unpack a list of pairs into two tuples

plt.plot(xf, yf)
plt.xlabel('Height in inches')
plt.ylabel('Probability')
plt.title('Conditional probability of being female')
plt.show()

#fit a linear regression to the data
reg_female = LinearRegression().fit(train['Height'].reshape(-1,1), train['Gender_Female'])

print('Intercept: %f' %reg_female.intercept_)
print('Coefficient: %f' %reg_female.coef_)

#predict female if conditional probability estimated by regression is above 50%
y_female = reg_female.predict(test['Height'].values.reshape(-1,1))

y_binary = []
for y in y_female:
	if y > 0.5:
		y_binary.append(1)
	else:
		y_binary.append(0)

#evaluate model
cm_regression = {'P = Male, A = Male':0,'P = Male, A = Female':0,'P = Female, A = Male':0,'P = Female, A = Female':0}
for i in range(5000):#dummy variables coded in inverse order
	if y_binary[i]==1 and y_test_dummy[i]==0:
		cm_regression['P = Female, A = Female']+=1
	elif y_binary[i]==1 and y_test_dummy[i]==1:
		cm_regression['P = Female, A = Male']+=1
	elif y_binary[i]==0 and y_test_dummy[i]==0:
		cm_regression['P = Male, A = Female']+=1
	else:
		cm_regression['P = Male, A = Male']+=1

print(cm_regression)		

print('Accuracy: %f' %((cm_regression['P = Female, A = Female']+cm_regression['P = Male, A = Male'])/5000))

recall = cm_regression['P = Female, A = Female']/(cm_regression['P = Female, A = Female']+cm_regression['P = Male, A = Female'])
precision = cm_regression['P = Female, A = Female']/(cm_regression['P = Female, A = Female']+cm_regression['P = Female, A = Male'])

print('Recall = %f' %recall)
print('Precision = %f' %precision)
print('F1 score = %f' %(2*(precision*recall)/(recall+precision)))

#this is the best model so far

#fit a logistic regression to the data
log_reg = LogisticRegression().fit(train['Height'].values.reshape(-1,1), train['Gender_Female'])

#obtain the predictions
log_y = log_reg.predict(test['Height'].values.reshape(-1,1))

#evaluate model
cm_log_reg = {'P = Male, A = Male':0,'P = Male, A = Female':0,'P = Female, A = Male':0,'P = Female, A = Female':0}
for i in range(5000):#dummy variables coded in inverse order
	if log_y[i]==1 and y_test_dummy[i]==0:
		cm_log_reg['P = Female, A = Female']+=1
	elif log_y[i]==1 and y_test_dummy[i]==1:
		cm_log_reg['P = Female, A = Male']+=1
	elif log_y[i]==0 and y_test_dummy[i]==0:
		cm_log_reg['P = Male, A = Female']+=1
	else:
		cm_log_reg['P = Male, A = Male']+=1

print(cm_log_reg)		

print('Log Reg Accuracy: %f' %((cm_log_reg['P = Female, A = Female']+cm_log_reg['P = Male, A = Male'])/5000))

log_recall = cm_log_reg['P = Female, A = Female']/(cm_log_reg['P = Female, A = Female']+cm_log_reg['P = Male, A = Female'])
log_precision = cm_log_reg['P = Female, A = Female']/(cm_log_reg['P = Female, A = Female']+cm_log_reg['P = Female, A = Male'])

print('Log Reg Recall = %f' %log_recall)
print('Log Reg Precision = %f' %log_precision)
print('Log Reg F1 score = %f' %(2*(log_precision*log_recall)/(log_recall+log_precision)))

#similar performance to linear regression

#fit the k-nearest neighbours algorithm
knn = KNeighborsClassifier(n_neighbors=400).fit(train['Height'].values.reshape(-1,1), train['Gender_Female']) #using default k=5

#obtain the predictions
knn_y = knn.predict(test['Height'].values.reshape(-1,1))

#evaluate model
cm_knn = {'P = Male, A = Male':0,'P = Male, A = Female':0,'P = Female, A = Male':0,'P = Female, A = Female':0}
for i in range(5000):#dummy variables coded in inverse order
	if knn_y[i]==1 and y_test_dummy[i]==0:
		cm_knn['P = Female, A = Female']+=1
	elif knn_y[i]==1 and y_test_dummy[i]==1:
		cm_knn['P = Female, A = Male']+=1
	elif log_y[i]==0 and y_test_dummy[i]==0:
		cm_knn['P = Male, A = Female']+=1
	else:
		cm_knn['P = Male, A = Male']+=1

print(cm_knn)		

print('knn accuracy: %f' %((cm_knn['P = Female, A = Female']+cm_knn['P = Male, A = Male'])/5000))

knn_recall = cm_knn['P = Female, A = Female']/(cm_knn['P = Female, A = Female']+cm_knn['P = Male, A = Female'])
knn_precision = cm_knn['P = Female, A = Female']/(cm_knn['P = Female, A = Female']+cm_knn['P = Female, A = Male'])

print('Knn Recall = %f' %knn_recall)
print('Knn Precision = %f' %knn_precision)
print('Knn F1 score = %f' %(2*(knn_precision*knn_recall)/(knn_recall+knn_precision)))

#overall accuracy slightly better than logistic regression for k = 5 (0.841), some evidence of overfitting
#using k=400, overall accuracy is very similar to logistic regression, not enough flexibility, oversmoothing

#plot accuracy vs k to pick the best value of k
knn_accuracy = {}
for k in [3,5,7,10,12,15,20,25,30,35,45,50,75,100,150,200]:
	#fit the k-nearest neighbours algorithm
	knn = KNeighborsClassifier(n_neighbors=k).fit(train['Height'].values.reshape(-1,1), train['Gender_Female'])

	#obtain the predictions
	knn_y = knn.predict(test['Height'].values.reshape(-1,1))

	#evaluate model
	cm_knn = {'P = Male, A = Male':0,'P = Male, A = Female':0,'P = Female, A = Male':0,'P = Female, A = Female':0}
	for i in range(5000):#dummy variables coded in inverse order
		if knn_y[i]==1 and y_test_dummy[i]==0:
			cm_knn['P = Female, A = Female']+=1
		elif knn_y[i]==1 and y_test_dummy[i]==1:
			cm_knn['P = Female, A = Male']+=1
		elif log_y[i]==0 and y_test_dummy[i]==0:
			cm_knn['P = Male, A = Female']+=1
		else:
			cm_knn['P = Male, A = Male']+=1	
	knn_accuracy[k] = (cm_knn['P = Female, A = Female']+cm_knn['P = Male, A = Male'])/5000
	
list = sorted(knn_accuracy.items()) # sorted by key, return a list of tuples
x, y = zip(*list) # unpack a list of pairs into two tuples

plt.plot(x, y)
plt.xlabel('n_neighbours')
plt.ylabel('accuracy')
plt.title('Overall accuracy per n_neighbours')
plt.show()

#best value of k seems to be around 10, but better to select with cross-validation

#repeat previous but using cross-validation
#plot accuracy vs k to pick the best value of k
knn_accuracy = {}
for k in [3,5,7,10,12,15,20,25,30,35,45,50,75,100,150,200]:
	#fit the k-nearest neighbours algorithm
	#knn = KNeighborsClassifier(n_neighbors=k).fit(train['Height'].values.reshape(-1,1), train['Gender_Female'])

	#obtain the predictions
	knn_y = cross_val_predict(KNeighborsClassifier(n_neighbors=k),train['Height'].values.reshape(-1,1),y=train['Gender_Female'],cv=5)

	#evaluate model
	cm_knn = {'P = Male, A = Male':0,'P = Male, A = Female':0,'P = Female, A = Male':0,'P = Female, A = Female':0}
	for i in range(5000):#dummy variables coded in inverse order
		if knn_y[i]==1 and y_test_dummy[i]==0:
			cm_knn['P = Female, A = Female']+=1
		elif knn_y[i]==1 and y_test_dummy[i]==1:
			cm_knn['P = Female, A = Male']+=1
		elif log_y[i]==0 and y_test_dummy[i]==0:
			cm_knn['P = Male, A = Female']+=1
		else:
			cm_knn['P = Male, A = Male']+=1	
	knn_accuracy[k] = (cm_knn['P = Female, A = Female']+cm_knn['P = Male, A = Male'])/5000
	
list = sorted(knn_accuracy.items()) # sorted by key, return a list of tuples
x, y = zip(*list) # unpack a list of pairs into two tuples

plt.plot(x, y)
plt.xlabel('n_neighbours')
plt.ylabel('accuracy')
plt.title('Overall accuracy per n_neighbours with cross-validation')
plt.show()

#using cross-validation, the best k seems to be 12, but accuracy much lower now for all values of k

#Naive Bayes approach
male_mean = train.groupby('Gender')['Height'].mean()['Male']
male_std = train.groupby('Gender')['Height'].std()['Male']
female_mean = train.groupby('Gender')['Height'].mean()['Female']
female_std = train.groupby('Gender')['Height'].std()['Female']
female_p = train.groupby('Gender')['Height'].count()['Female']/5000

f_male = norm.pdf(test['Height'],loc=male_mean, scale=male_std)
f_female = norm.pdf(test['Height'],loc=female_mean, scale=female_std) 

p_hat_Bayes = f_female*female_p/(f_female*female_p+f_male*(1-female_p)) #Naive Bayes estimate of conditional probability
#if the sample is biased, instead of using the estimated female_p (prevalence), we can force it to be balanced (0.5)

plt.scatter(test['Height'], p_hat_Bayes)
plt.xlabel('Height')
plt.ylabel('Probability of being female')
plt.title('Naive Bayes estimates')
plt.show()