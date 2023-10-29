import pandas as pd

# for regression, we are using the first 5 columns of the dataset
data = pd.read_csv("normalised_data.csv")[[
    'Neuroticism (N-Score)',
    'Extroversion (E-Score)',
    'Openness (O-Score)',
    'Agreeableness (A-Score)',
    'Conscientiousness (C-Score)'
]]

# applying regularisation / standardisation
standardized_df = (data - data.mean()) / data.std()
standardized_df.to_csv('regularized_personality_dimensions.csv', index=False)

# each column now has mean 0 and standard deviation 1
print(standardized_df.mean())
print(standardized_df.std())
