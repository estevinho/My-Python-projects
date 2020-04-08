import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pylab as plt

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