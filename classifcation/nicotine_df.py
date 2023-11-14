import pandas as pd

dataset = pd.read_csv('../Dataset/normalised_data.csv')

predictors = ['Neuroticism (N-Score)', 'Extroversion (E-Score)', 'Openness (O-Score)', 'Agreeableness (A-Score)',
              'Conscientiousness (C-Score)', 'Impulsiveness (BIS-11)', 'Sensation (SS)',
              'Age_18-24', 'Age_25-34', 'Age_35-44', 'Age_45-54', 'Age_55-64', 'Age_65+',
              'Gender_F', 'Gender_M', 'Country_Australia', 'Country_Canada', 'Country_Ireland', 'Country_New Zealand',
              'Country_Other', 'Country_UK', 'Country_USA', 'Ethnicity_Asian', 'Ethnicity_Black', 'Ethnicity_Other',
              'Ethnicity_White', 'Ethnicity_White-Asian', 'Ethnicity_White-Black',
              'Education_A', 'Education_B', 'Education_C', 'Education_D', 'Education_E', 'Education_F',
              'Education_G', 'Education_H', 'Education_I']
predicates = ['Nicotine']
total_set = predictors + predicates
df = dataset[total_set]
