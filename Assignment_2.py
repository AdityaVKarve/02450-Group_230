#First, I'll split the dataset into training and testing sets. 
#I'll split it in a 70-30 ratio

#Load dataset
import pandas as pd
dataset = pd.read_csv('./Dataset/normalised_data.csv')
#Check dataset
print(dataset.describe())

#Split into train and test sets
from sklearn.model_selection import train_test_split
train, test = train_test_split(dataset, test_size = 0.2)
#Training values
training_predictors = train.loc[:,['Extroversion (E-Score)','Openness (O-Score)','Agreeableness (A-Score)','Conscientiousness (C-Score)']]
training_target = train.loc[:,['Neuroticism (N-Score)']]
#Testing values
testing_predictors = test.loc[:,['Extroversion (E-Score)','Openness (O-Score)','Agreeableness (A-Score)','Conscientiousness (C-Score)']]
testing_target = test.loc[:,['Neuroticism (N-Score)']]


#First, using regular linear regression and getting variance
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(training_predictors, training_target)
model_output = linreg.predict(testing_predictors)

#Getting R-squared, MSE, RMS error
from sklearn.metrics import r2_score
from sklearn import metrics

r_squared = r2_score(testing_target, model_output)
mse = metrics.mean_squared_error(testing_target, model_output)

print("Simple linear regression summary:\nMSE:{}\nR_Squared:{}".format(mse,r_squared))

#Using ridge regression
from sklearn.linear_model import Ridge

alpha_vals = []
start_val = 1e-5
for i in range(10):
    alpha_vals.append(start_val)
    start_val*=10

#Get the right alpha
import sys
min_mse = sys.maxsize
min_mse_alpha = -1
mse_plot = []
alpha_plot = []
for i in range(len(alpha_vals)):
    ridgereg = Ridge(alpha=alpha_vals[i])
    ridgereg.fit(training_predictors, training_target)
    y_pred = ridgereg.predict(testing_predictors)
    mse = metrics.mean_squared_error(testing_target,y_pred)
    mse_plot.append(mse)
    alpha_plot.append(str(alpha_vals[i]))
    if mse < min_mse:
        min_mse = mse
        min_mse_alpha = alpha_vals[i]

#plotting MSE vs alpha
import matplotlib.pyplot as plt

plt.plot(alpha_plot, mse_plot, label = 'MSE vs alpha')
plt.xlabel('Alpha')
plt.ylabel('MSE')
plt.show()

print("Minimum MSE recorded using ridge regression:{}.".format(min_mse))
print("Alpha for minimum MSE:{}".format(min_mse_alpha))

print("Performing k-fold cross validation with k = 10")
avg_mse = 0
for i in range(10):
    train, test = train_test_split(dataset, test_size = 0.1)
    #Testing values
    testing_predictors = test.loc[:,['Extroversion (E-Score)','Openness (O-Score)','Agreeableness (A-Score)','Conscientiousness (C-Score)']]
    testing_target = test.loc[:,['Neuroticism (N-Score)']]
    y_pred_ridge = ridgereg.predict(testing_predictors)
    mse_ridge = metrics.mean_squared_error(testing_target,y_pred_ridge)
    y_pred_slr = linreg.predict(testing_predictors)
    mse_slr = metrics.mean_squared_error(testing_target,y_pred_slr)
    avg_mse += mse_ridge

print("Average MSE for k=10 fold cross validation:{}".format(avg_mse/10))