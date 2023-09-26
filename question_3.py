''' solution for question 3 '''
import pandas as pd
import numpy as np
from matplotlib.pyplot import boxplot, xticks, ylabel, title, show, hist, figure, subplot, xlabel, ylabel, scatter, xlim, ylim, plot, legend, grid,subplots
from scipy.linalg import svd

##Load dataset
dataset = pd.read_csv('./Dataset/drug_consumption_cleaned.csv')

##List feautures. Currently looking at just Cannabis vs personality traits.
features = ['Neuroticism (N-Score)', 'Extroversion (E-Score)', 'Openness (O-Score)',
            'Agreeableness (A-Score)','Conscientiousness (C-Score)','Impulsiveness (BIS-11)',
            'Sensation (SS)']
target = ['Cannabis']
nominal_col_names = ['Age','Gender','Country','Ethnicity','Education']



# extracting y's out of df
y = dataset[target].to_numpy().flatten()
classNames = sorted(set(y))

# extracting x's out of the df
X = dataset[features].to_numpy()


# question 3.1 solution for finding outliers begins here
print('----- queston 3.1 detecting outliers ------')
boxplot(X)
xticks(range(1,len(features)+1),features)
ylabel('points')
title('Drug addiction  - boxplot')
#show()
print('----- ---------------------- ------')

#question 3.2 finding if any of the attributes are normally distributed
print('----- queston 3.2 normal dist or not ------')
X_normal_neuro = dataset['Neuroticism (N-Score)'].to_numpy()
nbins = 5
X_normal_sens = dataset['Sensation (SS)'].to_numpy()
# Plot the samples and histogram
figure(figsize=(12,4))
title('Normal distribution')
subplot(1,2,1)
hist(X_normal_neuro , bins=nbins, edgecolor='black')
xlabel('Neuroticism (N-Score)')
ylabel('Count')
subplot(1,3,3)
hist(X_normal_sens, bins=nbins, edgecolor='black')
xlabel('Sensation (SS)')
ylabel('Count')
#show()
print('----- ---------------------- ------')


# correlation study
print('----- queston 3.3 correlation matrix ------')
corr_df =  dataset[features+target].corr('pearson', numeric_only=True)
print(corr_df)
corr_df_max = corr_df[corr_df!=1].max(axis=0)
corr_df_min = corr_df[corr_df!=1].min(axis=0)
print('-------max correlation row wise----------')
print(corr_df_max)
print('-------min correlation row wise----------')
print(corr_df_min)
scatter(dataset['Sensation (SS)'], dataset['Impulsiveness (BIS-11)'], label='Scatter Plot', color='blue', marker='.')
xlim(-10,30)
ylim(-5,30) 
xlabel('Sensation (SS)')
ylabel('Impulsiveness (BIS-11)')
title('Scatter plot Sensation vs Impulsiveness ')
#show()
print('----- ---------------------- ------')

#PCA
print('------------------ PCA ----------------')
nominal_cols = dataset[nominal_col_names].copy(deep=True)
one_hot_encoding = pd.get_dummies(nominal_cols)
new_dataset =  dataset.drop(nominal_col_names+target,axis=1)
new_dataset = new_dataset.join(one_hot_encoding)
new_dataset.to_csv('processed_dataset.csv')
norm_data = pd.read_csv("normalised_data.csv")
X = norm_data.to_numpy()
# Subtract mean value from data
Y = X - np.ones((len(y),1))*X.mean(axis=0)

# PCA by computing SVD of Y
U,S,Vh = svd(Y,full_matrices=True)

# Compute variance explained by principal components
rho = (S*S) / (S*S).sum() 

threshold = 0.9

# Plot variance explained
figure()
plot(range(1,len(rho)+1),rho,'x-')
plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
plot([1,len(rho)],[threshold, threshold],'k--')
title('Variance explained by principal components')
xlabel('Principal component')
ylabel('Variance explained')
legend(['Individual','Cumulative','Threshold'])
grid()
show()
print(new_dataset)
print('----- ---------------------- ------')

# Project data onto considered principle components
print('------------------ PCA ----------------')
V = Vh.T   
# Project the centered data onto principal component space
Z = Y @ V
i = 0
j = 1
f = figure()
title('Addictiveness of cannabis: PCA')
#Z = array(Z)
print(y.flatten())
print(y.shape)
for c in range(y.max()):
    # select indices belonging to class c:
    
    class_mask = y==c
    plot(Z[class_mask,i], Z[class_mask,j], 'o', alpha=.5)
legend(classNames)
xlabel('PC{0}'.format(i+1))
ylabel('PC{0}'.format(j+1))
show()
print('----- ---------------------- ------')

# Direction of considered principle components
print('------------------ Direction of considered principle components ----------------')
pc1 = Vh[0][:2]
pc2 = Vh[1][:2]

# Create a figure and axis
fig, ax = subplots()

# Plot the vector as an arrow
ax.arrow(0, 0, pc1[0], pc1[1], head_width=0.1, head_length=0.2, fc='blue', ec='blue')
ax.arrow(0, 0, pc2[0], pc2[1], head_width=0.1, head_length=0.2, fc='red', ec='red')
legend(['PC1','PC2'])


# Add labels and title
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_title('Vector Plot')

# Show the plot
grid()
show()

print('----- ---------------------- ------')