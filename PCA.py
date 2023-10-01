import pandas as pd
import numpy as np
from matplotlib.pyplot import boxplot, xticks, ylabel, title, show, hist, figure, subplot, xlabel, ylabel, scatter, xlim, ylim, plot, legend, grid,subplots
from scipy.linalg import svd

dataset = pd.read_csv('./Dataset/drug_consumption_cleaned_normalised.csv')

print(len(dataset.columns))