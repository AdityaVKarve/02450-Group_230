import torch
import matplotlib.pyplot as plt
from sklearn import model_selection
from toolbox_02450 import train_neural_net, draw_neural_net
import numpy as np

from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid, plot)

def find_opt_h(h_list,X,y,K,M):
    # K-fold crossvalidation               
    CV = model_selection.KFold(K, shuffle=True)
    errors = np.empty((K,len(h_list)))
    errors_train = np.empty((K,len(h_list)))
    for (k, (train_index, test_index)) in enumerate(CV.split(X,y)):
        for counter, h in enumerate(h_list):
        #Parameters for neural network classifier
            n_hidden_units = h      # number of hidden units
            n_replicates = 1        # number of networks trained in each k-fold
            max_iter = 1000
            #Define the model
            model = lambda: torch.nn.Sequential(
                        torch.nn.Linear(M, n_hidden_units), #M features to n_hidden_units
                        torch.nn.Tanh(),   # 1st transfer function,
                        torch.nn.Linear(n_hidden_units, 1), # n_hidden_units to 1 output neuron
                        # no final tranfer function, i.e. "linear output"
                        )
            loss_fn = torch.nn.MSELoss() # notice how this is now a mean-squared-error loss

            print('Training model of type:\n\n{}\n'.format(str(model())))
            #errors = [] # make a list for storing generalizaition error in each loop
            print('\nCrossvalidation fold: {0}/{1}'.format(k+1,K))    

            # Extract training and test set for current CV fold, convert to tensors
            X_train = torch.Tensor(X[train_index,:])
            y_train = torch.Tensor(y[train_index]).reshape(-1,1)
            X_test = torch.Tensor(X[test_index,:])
            y_test = torch.Tensor(y[test_index]).reshape(-1,1)
            # Train the net on training data
            net, final_loss, learning_curve = train_neural_net(model,
                                                            loss_fn,
                                                            X=X_train,
                                                            y=y_train,
                                                            n_replicates=n_replicates,
                                                            max_iter=max_iter)
            

            # print('\n\tBest loss: {}\n'.format(final_loss))

            # Determine estimated class labels for test set
            y_test_est = net(X_test)

            # Determine errors and errors
            se = (y_test_est.float()-y_test.float())**2 # squared error
            mse = (sum(se).type(torch.float)/len(y_test)).data.numpy() #mean
            errors[k,counter]= mse # store error rate for current CV fold 
            errors_train[k,counter] = final_loss
    mean_errors_per_h = np.mean(errors, axis=0)
    mean_errors_per_h_train = np.mean(errors_train,axis=0)
    print(mean_errors_per_h, np.min(mean_errors_per_h), np.argmin(mean_errors_per_h))
    
    h_star = h_list[np.argmin(mean_errors_per_h)]
    figure(k, figsize=(12,8))
    subplot(1,2,2)
    title('Optimal h: {0}'.format(h_star))
    plot(h_list, mean_errors_per_h,'b.-',h_list,mean_errors_per_h_train,'r.-')
    xlabel('value of hidden units')
    ylabel('Loss')
    legend(['Validation error','Train error'])
    grid()
    show()
    #Define the model
    model = lambda: torch.nn.Sequential(
                torch.nn.Linear(M, h_star), #M features to n_hidden_units
                torch.nn.Tanh(),   # 1st transfer function,
                torch.nn.Linear(h_star, 1), # n_hidden_units to 1 output neuron
                # no final tranfer function, i.e. "linear output"
                )
    loss_fn = torch.nn.MSELoss() # notice how this is now a mean-squared-error loss

    net, _, _ = train_neural_net(model,
                                loss_fn,
                                X=X_train,
                                y=y_train,
                                n_replicates=n_replicates,
                                max_iter=max_iter)
    return (h_star, net)