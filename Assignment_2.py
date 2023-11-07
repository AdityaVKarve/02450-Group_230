#First, I'll split the dataset into training and testing sets. 


#Load dataset
import pandas as pd
from pandas import DataFrame
dataset = pd.read_csv('./Dataset/normalised_data.csv')

predictors = ['Extroversion (E-Score)','Openness (O-Score)','Agreeableness (A-Score)','Conscientiousness (C-Score)','Impulsiveness (BIS-11)','Sensation (SS)','Age_18-24','Age_25-34','Age_35-44','Age_45-54','Age_55-64','Age_65+','Gender_F','Gender_M','Alcohol_Frequent','Amphetamines_Frequent','Amyl_Frequent','Benzos_Frequent','Caffeine_Frequent','Chocolate_Frequent','Cocaine_Frequent','Crack_Frequent','Ecstasy_Frequent','Heroin_Frequent','Ketamine_Frequent','Legal_Highs_Frequent','LSD_Frequent','Methamphetamine_Frequent','Magic_Mushrooms_Frequent','Nicotine_Frequent','Semer_Frequent','Inhalants_Frequent']
predicates = ['Neuroticism (N-Score)']
total_set = predictors + predicates
dataframe_relevant = dataset[total_set]


import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(dataframe_relevant.corr(), cmap="BuGn_r")
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



