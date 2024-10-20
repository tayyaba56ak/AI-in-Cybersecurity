import pandas as pd
import numpy as np  
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score,precision_score
from sklearn.model_selection import train_test_split 

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

df = pd.read_csv('C:/Users/ASUS/Downloads/Credit_card/creditcard_2023.csv')
print(df.head()) 
df.drop_duplicates().any()
df.shape 

print(df.describe())

missing_value = df.isnull().sum 
print("Missing value:", missing_value)

class_counts = df['Class'].value_counts()
print(class_counts)

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.hist(df['Class'],20)
plt.show()

correlation_matrix = df.corr()
plt.figure(figsize= (12,8))
sns.headmap(correlation_matrix, annot = False, cmap ='coolwarm', linewidths = 0.5 )
plt.title('Correlation Heatmap')
plt.show() 

df = df.sample(frac=1, random_state= 42) 
df = df.head(1000) 

scaler = StandardScaler()
column_to_scale = df.columns[1:-2]
scaler.fit_transform(df[column_to_scale])

df =df.drop(['id'], axis =1)
print(" removed successfully ")

print("columns:" , df.columns) 

class_counts = df['Class'].value_counts()
print(class_counts) 

print("\nshape", df.shape)
df.shape 

df.to_csv('preprocessed_dataset.csv', index=False) 
dfp= pd.read_csv('preprocessed_dataset.csv')

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(dfp.iloc[:, :-1], dfp.iloc[:, -1], test_size=0.2, random_state=42)
print("Training set size:", X_train.shape)
print("Testing set size:", X_test.shape) 

