
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
import numpy as np 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import time




data =pd.read_csv('C:/Users/ASUS/Downloads/Credit_card/creditcard_2023.csv') 

print(data.head()) 

data= data.drop_duplicates()

data.describe()
data.shape 
data.Class.value_counts() 

columns_to_scale = data.columns[1:-2]
scalar = StandardScaler()

data[columns_to_scale] = scalar.fit_transform(data[columns_to_scale])
print('Mean after standardization:')
print(scalar.mean_)

print('\n Standard deviation after standardization')
print(np.sqrt(scalar.var_))

print('\data after standardization')
print(data) 

data1 = data.drop([0], axis =0)
data1 = data1.drop(['id'], axis =1)
print(data1) 

x = data.drop('Class', axis =1) 
y =data.iloc[:,1] 
x_train, y_train, x_test, y_test = train_test_split(x, y, test_size= 0.2, random_state=42)

print('Train set size:', x_train.shape)
print('Test set size:', x_test.shape) 

plt.hist(data['Class'], bins=[0,0.1,0.9,1.0], edgecolor = 'r', color = 'r') 
plt.show()

data_correlation = data.corr()
sns.heatmap(data_correlation, cmap = 'viridis') 
plt.title('Correlation')
plt.show() 

clf =LinearDiscriminantAnalysis()
start_time = time.time() 
clf.fit(x_train, y_train) 
end_time = time.time() 

training_time = end_time-start_time
print('training time:' , training_time) 

start_time =time.time()
y_pred = clf.predict(x_test)
end_time = time.time() 

prediction_time = end_time-start_time 
print(x_test.iloc[:2, :2])
print(x_test.shape)
print(y_pred.shape) 

print('prediction time:' , training_time) 
accuracy = clf.score(x_train, y_test)
print('accuracy:' , accuracy) 

print(y_train.shape)
accuracy1=accuracy_score(y_test,y_pred) #y_test and y_pred should have the same length
precision1=precision_score(y_test,y_pred,average='weighted')
recall1=recall_score(y_test,y_pred,average='weighted')
fscore1=f1_score(y_test,y_pred,average='weighted')
print(recall1,precision1,accuracy1,f1_score) 

print(classification_report(y_test,y_pred))


cm=confusion_matrix(y_test,y_pred)
print(y_test.iloc[:].value_counts())
disp=ConfusionMatrixDisplay( confusion_matrix=cm)
disp.plot()
plt.title('Confusion Matrix')
  

    

 




  
    
