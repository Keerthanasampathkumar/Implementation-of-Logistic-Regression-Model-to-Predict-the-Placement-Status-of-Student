# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries.

2. Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.

3. Import LabelEncoder and encode the dataset.

4. Import LogisticRegression from sklearn and apply the model on the dataset.

5. Predict the values of array.

6. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.

7. Apply new unknown values

## Program:
```c
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: KEERTHANA S
RegisterNumber:  212222230066

#import modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#reading the file
dataset = pd.read_csv('Placement_Data_Full_Class.csv')
dataset

dataset.head(20)

dataset.tail(20)

#droping tha serial no salary col
dataset = dataset.drop('sl_no',axis=1)
#dataset = dataset.drop('salary',axis=1)

dataset = dataset.drop('gender',axis=1)
dataset = dataset.drop('ssc_b',axis=1)
dataset = dataset.drop('hsc_b',axis=1)
dataset

dataset.shape

dataset.info()

#catgorising col for further labelling
dataset["degree_t"] = dataset["degree_t"].astype('category')
dataset["workex"] = dataset["workex"].astype('category')
dataset["specialisation"] = dataset["specialisation"].astype('category')
dataset["status"] = dataset["status"].astype('category')
dataset["hsc_s"] = dataset["hsc_s"].astype('category')
dataset.dtypes

dataset.info()

dataset["degree_t"] = dataset["degree_t"].cat.codes
dataset["workex"] = dataset["workex"].cat.codes
dataset["specialisation"] = dataset["specialisation"].cat.codes
dataset["status"] = dataset["status"].cat.codes
dataset["hsc_s"] = dataset["hsc_s"].cat.codes
dataset

dataset.info()

dataset

#selecting the features and labels
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x, y,test_size=0.2)
dataset.head()

x_train.shape

x_test.shape

y_train.shape

y_test.shape

from sklearn.linear_model import LogisticRegression
clf= LogisticRegression()
clf.fit(x_train,y_train)
clf.score(x_test,y_test)

clf.predict([[0, 87, 0, 95, 0, 2, 78, 2, 0]])
```

## Output:
### DATASET:
![image](https://github.com/Keerthanasampathkumar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119477890/db2c900c-a706-4f13-85a4-03f3a2db4c44)
### dataset.head():
![image](https://github.com/Keerthanasampathkumar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119477890/aac28701-1f20-401a-8a01-92c6591e10ce)
### dataset.tail():
![image](https://github.com/Keerthanasampathkumar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119477890/441c46ec-bd59-4cb3-afbf-c79eff42d77c)
### dataset after dropping:
![image](https://github.com/Keerthanasampathkumar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119477890/ed6e1391-7e7b-465c-a5a5-0fa64fb6862e)
![image](https://github.com/Keerthanasampathkumar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119477890/870d0b43-9dc5-4649-8f0e-931dd511c99d)
### datase.shape:
![image](https://github.com/Keerthanasampathkumar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119477890/9831900c-a273-490a-97d2-4eb261863bd9)
### dataset.info()
![image](https://github.com/Keerthanasampathkumar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119477890/90227390-a813-4d38-abd1-a994205977f9)
### dataset.dtypes:
![image](https://github.com/Keerthanasampathkumar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119477890/95c7358d-7fb3-4f32-b15c-c387533cf022)
### dataset.info():
![image](https://github.com/Keerthanasampathkumar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119477890/d197c21d-b516-46e3-86fc-b7a99e535653)
### dataset.codes:
![image](https://github.com/Keerthanasampathkumar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119477890/6ce99cf2-c711-49f2-b6e7-4739829d13e0)
### selecting the features and labels:
![image](https://github.com/Keerthanasampathkumar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119477890/548053fe-6ca8-44ec-8fdb-70ffe76bed69)
### dataset.head():
![image](https://github.com/Keerthanasampathkumar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119477890/578ff7d2-4d4a-403e-b949-0305899bcba3)
### x_train.shape:
![image](https://github.com/Keerthanasampathkumar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119477890/ce7fbb17-d5f5-4970-b122-c775864f66d4)
### x_test.shape:
![image](https://github.com/Keerthanasampathkumar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119477890/c08b27cf-eaff-4db5-9130-2be9da7e7c12)
### y_train.shape:
![image](https://github.com/Keerthanasampathkumar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119477890/bf74725a-48b0-46c4-ab03-09296f1c3034)
### y_test.shape:
![image](https://github.com/Keerthanasampathkumar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119477890/ef2ca366-b3f2-4dab-a555-0682ff2a0f49)
![image](https://github.com/Keerthanasampathkumar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119477890/dd2d9497-5d24-4c8d-9ecb-faec028c0525)
### clf.predict:
![image](https://github.com/Keerthanasampathkumar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119477890/cb1ff9f9-86b0-43e0-97e0-9088e88fcc17)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
