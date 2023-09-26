import pandas as pd


'''data = pd.read_csv("processed_dataset.csv")
for i in range(len(data['Neuroticism (N-Score)'])):
    data['Neuroticism (N-Score)'].iloc[i] = (data['Neuroticism (N-Score)'].iloc[i] - min(data['Neuroticism (N-Score)']))/(max(data['Neuroticism (N-Score)']) - min(data['Neuroticism (N-Score)']))
    data['Extroversion (E-Score)'].iloc[i] = (data['Extroversion (E-Score)'].iloc[i] - min(data['Extroversion (E-Score)']))/(max(data['Extroversion (E-Score)']) - min(data['Extroversion (E-Score)']))
    data['Openness (O-Score)'].iloc[i] = (data['Openness (O-Score)'].iloc[i] - min(data['Openness (O-Score)']))/(max(data['Openness (O-Score)']) - min(data['Openness (O-Score)']))
    data['Agreeableness (A-Score)'].iloc[i] = (data['Agreeableness (A-Score)'].iloc[i] - min(data['Agreeableness (A-Score)']))/(max(data['Agreeableness (A-Score)']) - min(data['Agreeableness (A-Score)']))
    data['Conscientiousness (C-Score)'].iloc[i] = (data['Conscientiousness (C-Score)'].iloc[i] - min(data['Conscientiousness (C-Score)']))/(max(data['Conscientiousness (C-Score)']) - min(data['Conscientiousness (C-Score)']))

data.to_csv('normalised_data.csv',index=False)

'''
