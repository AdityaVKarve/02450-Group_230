###
#Performing PCA analysis on data for now
#The purpose is to aid in data visualisation
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt

##Load dataset
dataset = pd.read_csv('./Dataset/drug_consumption_cleaned.csv')

##List feautures. Currently looking at just Cannabis vs personality traits.
features = ['Neuroticism (N-Score)','Extroversion (E-Score)','Openness (O-Score)','Agreeableness (A-Score)','Conscientiousness (C-Score)','Impulsiveness (BIS-11)','Sensation (SS)']
target = ['Cannabis']
x = StandardScaler().fit_transform(dataset.loc[:,features].values)
y = StandardScaler().fit_transform(dataset.loc[:,target].values)

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents, columns = ['PCA 1', 'PCA 2'])

finalDf = pd.concat([principalDf,dataset[target]], axis = 1)


##PCA variance
variance = pca.explained_variance_ratio_
print(variance)


#Plot

plt.figure(figsize=(10,10))
plt.xticks(fontsize=12)
plt.yticks(fontsize=14)
plt.xlabel('Principal Component - 1',fontsize=20)
plt.ylabel('Principal Component - 2',fontsize=20)
plt.title("Principal Component Analysis of Cannabis dependency based on personality traits.",fontsize=20)
targets = [0,4]
colors = ['blue', 'orange','green','red','purple','brown','pink']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['Cannabis'] == target
    plt.scatter(finalDf.loc[indicesToKeep, 'PCA 1']
               , finalDf.loc[indicesToKeep, 'PCA 2'], c = color, s = 50)

plt.legend(targets,prop={'size': 15})
plt.show()