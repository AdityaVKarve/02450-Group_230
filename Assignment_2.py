#First, I'll split the dataset into training and testing sets. 


#Load dataset
import pandas as pd
from pandas import DataFrame
dataset = pd.read_csv('./Dataset/normalised_data.csv')

predictors = ['Neuroticism (N-Score)','Extroversion (E-Score)','Openness (O-Score)','Agreeableness (A-Score)']
predicates = ['Conscientiousness (C-Score)']
total_set = predictors + predicates
dataframe_relevant = dataset[total_set]


import seaborn as sns
import matplotlib.pyplot as plt

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



#trying ridge regression
print("Attempting ridge regression:")

alpha_val = 1e-5
best_alpha = -1
base_r2 = -123213213
base_mse = 123123123
mse_plot = []
alpha_plot = []
for i in range(10):
    print("Current alpha: {}".format(alpha_val))
    reg = linear_model.Ridge(alpha=alpha_val)
    reg.fit(X_train,y_train)

    y_pred = reg.predict(X_test)

    output_df = pd.DataFrame(y_pred, columns=predicates)

    mse = mean_squared_error(output_df,y_test)
    r_2 = r2_score(output_df,y_test)
    if r_2 > base_r2:
        base_r2 = r_2
        base_mse = mse
        best_alpha = alpha_val
    mse_plot.append(mse)
    alpha_plot.append(str(alpha_val))
    
    alpha_val *= 10
        
print("Best val of alpha:{}\nR_2:{}\nMSE:{}".format(best_alpha,base_r2,base_mse))
plt.plot(alpha_plot, mse_plot)
plt.xlabel("Alpha")
plt.ylabel("MSE")
plt.show()