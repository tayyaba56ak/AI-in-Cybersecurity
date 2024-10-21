import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier  
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt 
from sklearn.metrics import classification_report, accuracy_score,confusion_matrix 
import seaborn as sns


df = pd.read_csv('C:/Users/ASUS/Downloads/Credit_card/creditcard_2023.csv')

print(df.head()) 
print("data inof: ", df.info())
df.drop_duplicates() 
df.shape
df.describe() 

missing_values = df.isnull().sum().any()
print("Missing values: " , missing_values) 

class_count = df['Class'].value_counts()
print(class_count) 

fig = plt.figure() 
ax = fig.add_axes( [0,0,1,1])
ax.hist(df['Class'],20) 
plt.show() 

correlation_matrix = df.corr() 
plt.figure(figsize=(12 , 8))
sns.heatmap(correlation_matrix, annot = False, cmap= 'coolwarm',linewidths = 0.5 )
plt.title("Correlation heatmap")
plt.show() 
 
df = df.sample(frac=1, random_state= 42) 
df = df.head(1000)

column_to_scale = df.column[1:-2]
scaler = StandardScaler()
scaler.fit_transform(df[column_to_scale]) 

data = df.drop(['id'], axis= 1)
print(" the id removed  successfully") 

df.to_csv('preprocessed_dataset.csv', index=False)
dfp= pd.read_csv('preprocessed_dataset.csv') 

X_train, X_test, y_train, y_test = train_test_split( dfp.iloc[:, :-1], test_size= 0.3 , random_state= 42)

print("Training set size:", X_train.shape)
print("Testing set size:", X_test.shape) 


df= pd.read_csv('preprocessed_dataset.csv')
sns.countplot(x = df["Class"]) 

X = df.drop(['Class'], axis= 1)
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Print the size of each set
print("Training set size:", X_train.shape)
print("Testing set size:", X_test.shape)

#lets scale the data
scaler = StandardScaler()
scaled_feature_train = scaler.fit_transform(X_train) #prevent data leakage
scaled_feature_test = scaler.transform(X_test)

# Logistic Regression
lr=LogisticRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test) 

def model_eval(actual, predicted): 
    acc_score = accuracy_score(actual,predicted)
    conf_matrix = confusion_matrix(actual, predicted)
    class_rep = classification_report(actual, predicted) 

    print(" Model accuracy is : ", round(acc_score, 2))
    print(conf_matrix)
    print(class_rep) 

preds_lr_train = lr.predict(X_train)
preds_lr_test = lr.predict(X_test) 
print('Training Accuracy')
model_eval(y_train,preds_lr_train) 
 
print('Test Accuracy')
model_eval(y_test, preds_lr_test) 

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 6))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='cividis', linewidths=0.4, square=True, cbar=True,
    xticklabels=["0", "1"],
    yticklabels=["0", "1"]
)

plt.xlabel('Predicted', fontsize=14, fontweight='bold')
plt.ylabel('Actual', fontsize=14, fontweight='bold')
plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
plt.yticks(rotation=360)
plt.show() 

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
yy_pred = rf.predict(X_test) 
preds_rf_train = rf.predict(X_train)
preds_rf_test = rf.predict(X_test)
print('Training Accuracy')
model_eval(y_train, preds_rf_train)
print('Test Accuracy')
model_eval(y_test, preds_rf_test) 

cm = confusion_matrix(y_test, yy_pred)

plt.figure(figsize=(10, 6))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='cividis', linewidths=0.4, square=True, cbar=True,
    xticklabels=["0", "1"],
    yticklabels=["0", "1"]
)

plt.xlabel('Predicted', fontsize=14, fontweight='bold')
plt.ylabel('Actual', fontsize=14, fontweight='bold')
plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
plt.yticks(rotation=360) 
plt.show()
