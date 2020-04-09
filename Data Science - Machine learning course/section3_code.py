import pandas as pd
import statistics as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#load data
data = pd.read_csv('Galton.csv')

#keep male only
male_data = data[data['sex']=='M']

#generate test and training sets
male_train, male_test = train_test_split(male_data, test_size=0.5)

#guess prediction
avg_height = male_train['height'].mean()
print(avg_height)

#calculate mean squared error loss for guess prediction
v1 = []
for i in male_test['height'].values.tolist():
	loss = (i-avg_height)**2
	v1.append(loss)

print(st.mean(v1))
	
#fit linear regression
reg = LinearRegression().fit(male_train['father'].reshape(-1,1), male_train['height'])

print('Intercept: %f' %reg.intercept_)
print('Coefficient: %f' %reg.coef_)

#calculate mean squared error loss
height_predict = reg.predict(male_test['father'].reshape(-1,1))
print(mean_squared_error(male_test['height'],height_predict))

#loss is slightly lower, so prediction from linear regression is slightly better than guess