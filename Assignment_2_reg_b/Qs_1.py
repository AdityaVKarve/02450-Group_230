
# exercise 8.1.1
from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid,subplots_adjust,subplots)
import numpy as np, scipy.stats as st
import pandas as pd
from scipy.io import loadmat
import sklearn.linear_model as lm
from sklearn import model_selection
from toolbox_02450 import rlr_validate
from reg_net import find_opt_h
import torch

predictors = ['Neuroticism (N-Score)','Extroversion (E-Score)','Openness (O-Score)','Agreeableness (A-Score)']
predicates = ['Conscientiousness (C-Score)']

mat_data = pd.read_csv('../Dataset/normalised_data.csv')

X = mat_data[predictors].to_numpy()
y = mat_data[predicates].to_numpy().squeeze()
attributeNames = [predictors]
N, M = X.shape

# Add offset attribute
X = np.concatenate((np.ones((X.shape[0],1)),X),1)
attributeNames = [u'Offset']+attributeNames
M = M+1

## Crossvalidation
# Create crossvalidation partition for evaluation
K = 10
CV = model_selection.KFold(K, shuffle=True)
#CV = model_selection.KFold(K, shuffle=False)

# Values of lambda
lambdas = np.power(10.,range(-5,+5))

# Initialize variables
#T = len(lambdas)
Error_train_rlr = np.empty((K,1))
Error_test_rlr = np.empty((K,1))
Error_train_nofeatures = np.empty((K,1))
Error_test_nofeatures = np.empty((K,1))
w_rlr = np.empty((M,K))
mu = np.empty((K, M-1))
sigma = np.empty((K, M-1))
w_noreg = np.empty((M,K))

k=0
ann_errors = []
best_h = []
opt_lambda_errors = []
best_lambdas = []
for train_index, test_index in CV.split(X,y):
    print(train_index)
    # extract training and test set for current CV fold
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    internal_cross_validation = 10    
    
    opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X_train, y_train, lambdas, internal_cross_validation)
    
    best_lambdas.append(opt_lambda)

    Xty = X_train.T @ y_train
    XtX = X_train.T @ X_train
    
    # Compute mean squared error without using the input data at all
    Error_train_nofeatures[k] = np.square(y_train-y_train.mean()).sum(axis=0)/y_train.shape[0]
    Error_test_nofeatures[k] = np.square(y_test-y_test.mean()).sum(axis=0)/y_test.shape[0]

    # Estimate weights for the optimal value of lambda, on entire training set
    lambdaI = opt_lambda * np.eye(M)
    lambdaI[0,0] = 0 # Do no regularize the bias term
    w_rlr[:,k] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
    # Compute mean squared error with regularization with optimal lambda
    Error_train_rlr[k] = np.square(y_train-X_train @ w_rlr[:,k]).sum(axis=0)/y_train.shape[0]
    Error_test_rlr[k] = np.square(y_test-X_test @ w_rlr[:,k]).sum(axis=0)/y_test.shape[0]
    opt_lambda_errors.append(Error_test_rlr[k])

    #Find Optimal h for ANN using 10 fold cross validation 
    h_star, net = find_opt_h([1,2,3,4,5,6],X_train,y_train,K,M)
    print(h_star)

    # Determine estimated class labels for test set
    X_test_tensor = torch.Tensor(X_test)
    y_test_est = net(X_test_tensor)
    y_test_tensor = torch.Tensor(y_test).reshape(-1,1)
    # Determine errors and errors
    se = (y_test_est.float()-y_test_tensor.float())**2 # squared error
    mse = (sum(se).type(torch.float)/len(y_test_tensor)).data.numpy() #mean
    ann_errors.append(mse) # store error rate for current CV fold    
    best_h.append(h_star)

    # Display the results for the last cross-validation fold
    if k == K-1:
        figure(k, figsize=(10,10))
        subplot(1,2,1)
        semilogx(lambdas,mean_w_vs_lambda.T[:,1:],'.-') # Don't plot the bias term
        xlabel('Regularization factor')
        ylabel('Mean Coefficient Values')
        grid()
        # You can choose to display the legend, but it's omitted for a cleaner 
        # plot, since there are many attributes
        #legend(attributeNames[1:], loc='best')
        
        subplot(1,2,2)
        title('Optimal lambda: 1e{0}'.format(np.log10(opt_lambda)))
        loglog(lambdas,train_err_vs_lambda.T,'b.-',lambdas,test_err_vs_lambda.T,'r.-')
        xlabel('Regularization factor')
        ylabel('Squared error (crossvalidation)')
        legend(['Train error','Validation error'])
        grid()
        subplots_adjust(wspace=0.4)
    
    # To inspect the used indices, use these print statements
    #print('Cross validation fold {0}/{1}:'.format(k+1,K))
    #print('Train indices: {0}'.format(train_index))
    #print('Test indices: {0}\n'.format(test_index))

    k+=1

# show()
# # Display results
# print('Regularized linear regression:')
# print('- Training error: {0}'.format(Error_train_rlr.mean()))
# print('- Test error:     {0}'.format(Error_test_rlr.mean()))
# print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train_rlr.sum())/Error_train_nofeatures.sum()))
# print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum()-Error_test_rlr.sum())/Error_test_nofeatures.sum()))

# print('Weights in last fold:')
# for m in range(M):
#     print('{:>15} {:>15}'.format(attributeNames[m], np.round(w_rlr[m,-1],2)))

# print('Ran Exercise 8.1.1')

#Table of rlr,ANN,baseline gen errors with optimal h and lambda for each fold
fig, ax = subplots()
# Hide the axes
ax.axis('off')
data = np.empty((K,6))
for k in range(0,K):
    data[k][0] = k+1
    data[k][1] = best_h[k]
    data[k][2] = ann_errors[k]
    data[k][3] = best_lambdas[k]
    data[k][4] = opt_lambda_errors[k]
    data[k][5] = Error_test_nofeatures[k]

# header_row = ["Outer fold", "ANN", "Linear regression","Baseline"]
header_row_above = ['i', 'h*','$E^{\mathrm{test}}_i$(ANN)' ,'λ*','$E^{\mathrm{test}}_i$(Linear regression)','$E^{\mathrm{test}}_i$(Baseline)']
data = np.vstack([header_row_above, data])
# Create the table and add data
table = ax.table(cellText=data, colLabels=None,
                 loc='center', cellLoc='center', colLoc='center')

# Set the font size for the table
table.auto_set_font_size(False)
table.set_fontsize(12)

# Manually set column widths
col_widths = [0.2] * data.shape[1] 
table.auto_set_column_width(col=list(range(data.shape[1])))
# table.scale(1, 1.5)

# Display the table
show()

#Section 3
alpha = 0.05
#Regularized linear regression vs ANN
#Doing z = zA - zB one each of the folds
z =  np.array(opt_lambda_errors)-np.array(ann_errors) 
CI = st.t.interval(1-alpha, len(z)-1, loc=np.mean(z), scale=st.sem(z))  # Confidence interval
p = 2*st.t.cdf( -np.abs( np.mean(z) )/st.sem(z), df=len(z)-1)  # p-value 
print("*****************ANN vs regularized linear regression*****************") 
print("CI: ",CI)
print("p: ",p)
# ANN vs baseline
print("*****************ANN vs baseline *****************") 
z =  np.array(Error_test_nofeatures)-np.array(ann_errors) 
CI = st.t.interval(1-alpha, len(z)-1, loc=np.mean(z), scale=st.sem(z))  # Confidence interval
p = 2*st.t.cdf( -np.abs( np.mean(z) )/st.sem(z), df=len(z)-1)  # p-value  
print("CI: ",CI)
print("p: ",p)

#Regularized linear regression vs baseline
print("*****************regularized linear regression vs baseline*****************") 
z =  np.array(Error_test_nofeatures) - np.array(opt_lambda_errors) 
CI = st.t.interval(1-alpha, len(z)-1, loc=np.mean(z), scale=st.sem(z))  # Confidence interval
p = 2*st.t.cdf( -np.abs( np.mean(z) )/st.sem(z), df=len(z)-1)  # p-value  
print("CI: ",CI)
print("p: ",p)
