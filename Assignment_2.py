#First, I'll split the dataset into training and testing sets. 


#Load dataset
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import cross_val_score
import seaborn as sns
import matplotlib.pyplot as plt


dataset = pd.read_csv('./Dataset/normalised_data.csv')

predictors = ['Neuroticism (N-Score)','Extroversion (E-Score)','Openness (O-Score)','Agreeableness (A-Score)']
predicates = ['Conscientiousness (C-Score)']
total_set = predictors + predicates
dataframe_relevant = dataset[total_set]




#plt.show()

corr_df = dataframe_relevant.corr()
corr_df.to_csv('Correlation_matrix.csv')


#Splitting data into training and test splits

predictor_total_set = dataframe_relevant[predictors]
predicates_total_set = dataframe_relevant[predicates]

# split into train test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(predictor_total_set, predicates_total_set, test_size=0.33)

#Simple linear regression
print("Attempting simple linear regression:")
from sklearn import linear_model

reg = linear_model.LinearRegression()
reg.fit(X_train,y_train)

y_pred = reg.predict(X_test)

output_df = pd.DataFrame(y_pred, columns=predicates)

#Get Rsq and MSE for SLR 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score 
for i in output_df.columns:
    print("MSE for column {}: {}".format(i,mean_squared_error(output_df[i],y_test[i])))
    print("R^2 for column {}: {}".format(i,r2_score(output_df[i],y_test[i])))
    print("______________________________")



#Feature normalisation: Setting mean to 0 and stdev to 1
for c in predicates_total_set.columns:
    predicates_total_set[c] = predicates_total_set[c].sub(predicates_total_set[c].mean())
    predicates_total_set[c] = predicates_total_set[c].div(predicates_total_set[c].std())

for c in predictor_total_set.columns:
    predictor_total_set[c] = predictor_total_set[c].sub(predictor_total_set[c].mean())
    predictor_total_set[c] = predictor_total_set[c].div(predictor_total_set[c].std())




#trying ridge regression
print("Attempting ridge regression:")

alpha_val = 1e-5
best_alpha = -1
base_r2 = -123213213
base_mse = 123123123
accuracy_mean = []
accuracy_std = []
alpha_plot = []
for i in range(10):
    print("Current alpha: {}".format(alpha_val))
    print("Trying k-fold cross validation")
    ridge = linear_model.Ridge(alpha=alpha_val)
    scores = cross_val_score(ridge, predictor_total_set, predicates_total_set, cv=10)
    print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
    alpha_val*=10
    accuracy_mean.append(scores.mean())
    accuracy_std.append(scores.std())
    alpha_plot.append(str(alpha_val))
        
print("Best val of alpha:{}\nR_2:{}\nMSE:{}".format(best_alpha,base_r2,base_mse))
plt.plot(alpha_plot, accuracy_mean)
plt.xlabel("Alpha")
plt.ylabel("Accuracy (mean)")
plt.show()


plt.plot(alpha_plot, accuracy_std)
plt.xlabel("Alpha")
plt.ylabel("Accuracy (stdev)")
plt.show()

#For lambda of 10000
ridge = linear_model.Ridge(alpha=0)
ridge.fit(X_train,y_train)

df = pd.DataFrame(zip(X_train.columns, ridge.coef_))
df.to_csv("coeff.csv")
print(ridge.intercept_)

#Logistic regression
from sklearn.linear_model import LogisticRegression
dataset = pd.read_csv('./Dataset/normalised_data.csv')

predictors = ['Neuroticism (N-Score)', 'Extroversion (E-Score)', 'Openness (O-Score)', 'Agreeableness (A-Score)',
              'Conscientiousness (C-Score)', 'Impulsiveness (BIS-11)', 'Sensation (SS)',
              'Age_18-24', 'Age_25-34', 'Age_35-44', 'Age_45-54', 'Age_55-64', 'Age_65+',
              'Gender_F', 'Gender_M', 'Country_Australia', 'Country_Canada', 'Country_Ireland', 'Country_New Zealand',
              'Country_Other', 'Country_UK', 'Country_USA', 'Ethnicity_Asian', 'Ethnicity_Black', 'Ethnicity_Other',
              'Ethnicity_White', 'Ethnicity_White-Asian', 'Ethnicity_White-Black',
              'Education_A', 'Education_B', 'Education_C', 'Education_D', 'Education_E', 'Education_F',
              'Education_G', 'Education_H', 'Education_I']
predicates = ['Nicotine_Frequent']
total_set = predictors + predicates
predicates_total_set = dataset[predicates]
predictor_total_set = dataset[predictors]


logReg = LogisticRegression(solver='saga', penalty= 'elasticnet',random_state=0, l1_ratio=0)
scores = cross_val_score(logReg, predictor_total_set, predicates_total_set, cv=10)
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

alpha_val = 1e-5
best_alpha = -1
base_r2 = -123213213
base_mse = 123123123
accuracy_mean = []
accuracy_std = []
alpha_plot = []
for i in range(10):
    print("Current alpha: {}".format(alpha_val))
    print("Trying k-fold cross validation")
    log = LogisticRegression(solver='saga', penalty= 'l1',random_state=1, l1_ratio=alpha_val, class_weight = 'balanced')
    scores = cross_val_score(log, predictor_total_set, predicates_total_set, cv=10)
    print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
    alpha_val*=10
    accuracy_mean.append(scores.mean())
    accuracy_std.append(scores.std())
    alpha_plot.append(str(alpha_val))

print("Best val of alpha:{}\nR_2:{}\nMSE:{}".format(best_alpha,base_r2,base_mse))
plt.plot(alpha_plot, accuracy_mean)
plt.xlabel("Alpha")
plt.ylabel("Accuracy (mean)")
plt.show()


plt.plot(alpha_plot, accuracy_std)
plt.xlabel("Alpha")
plt.ylabel("Accuracy (stdev)")
plt.show()