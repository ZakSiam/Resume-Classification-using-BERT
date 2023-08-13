# Importing all the required packages or libraries
import numpy as np 
import pandas as pd

# Setting the seed value for reproducibility
seed_val = 42
np.random.seed(seed_val)

# Loading the Resume.csv dataset
file_path = './data/Resume.csv'
data = pd.read_csv(file_path)
data = data[['Resume_str', 'Category']]

# Factorize the categories in the dataset and then save the mapping for further use in script.py
y, categories_mapping = data['Category'].factorize()
categories_mapping = pd.Series(categories_mapping)
categories_mapping.to_csv('./categories_mapping.csv', index=False, header=['Category'])
