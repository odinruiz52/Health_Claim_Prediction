import pandas as pd

# Load the encoded dataset
df = pd.read_csv('insurance_encoded.csv')

# Get the number of rows (patients) in the dataset
num_patients = df.shape[0]
print(f'The number of patients in the dataset is: {num_patients}')
