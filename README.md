![Screenshot 2025-05-14 094055](https://github.com/user-attachments/assets/fc5f426c-0d8c-4705-8524-c6ebda87049b)![Screenshot 2025-05-14 094055](https://github.com/user-attachments/assets/6a39f9b4-04ff-4a39-ace6-439ffc2adcf4)![Screenshot 2025-05-14 094055](https://github.com/user-attachments/assets/9b1e56a1-2936-4c17-88ec-a8b11bd0c5ac)![Screenshot 2025-05-14 094055](https://github.com/user-attachments/assets/c9ee0bea-4513-44bb-83e0-0fe0f91c72cd)![Screenshot 2025-05-14 094055](https://github.com/user-attachments/assets/de5d9993-eb2b-4150-82ab-62efc4f2bd9b)# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1. Load & Preprocess: Read placement CSV, drop "sl_no", "salary", encode categorical columns.
2. Split Data: Extract features (X) and target (y), split into train/test (20% test).
3. Train & Predict: Train Logistic Regression model, predict test set outcomes.
4. Evaluate: Compute accuracy, confusion matrix, print classification report, predict sample.
```

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: VIMALRAJ B
RegisterNumber: 212224230304
*/

import pandas as pd
data=pd.read_csv('/content/Placement_Data.csv')
data.head()
data1=data.copy()
data1 = data1.drop(["sl_no", "salary"], axis = 1)#removes the specified row or column
data1.head()
data1.isnull().sum()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1
x=data1.iloc[:,:-1]
x
y=data1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear") # A Library for Large Linear Classification
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred) #Accuracy Score = (TP+TN)/ (TP+FN+TN+FP) , True
#accuracy_score(y_true, y_pred, normalize=False)
#Normalize : It contains the boolean value(True/False).If False, return the number of cor
#Otherwise, it returns the fraction of correctly confidential samples.
accuracy
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion #11+24=35 -correct predictions, 5+3=8 incorrect predictions
from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:
![Screenshot 2025-05-14 094055](https://github.com/user-attachments/assets/342db037-1e10-49d7-bfa0-700b828a646b)

![Screenshot 2025-05-14 094129](https://github.com/user-attachments/assets/8f2cf175-8179-45b8-9ba6-c226d8d034a8)

![Screenshot 2025-05-14 094153](https://github.com/user-attachments/assets/f4b9003e-737f-4528-888c-ab902af10a14)

![Screenshot 2025-05-14 094331](https://github.com/user-attachments/assets/d2e168f5-024a-4be7-868e-4d03398be6b6)

![Screenshot 2025-05-14 094345](https://github.com/user-attachments/assets/b74e5429-9324-4a46-b856-a72460a10c99)

![Screenshot 2025-05-14 094354](https://github.com/user-attachments/assets/4b8bb54c-524e-4ed4-8c4a-3f8a97076cce)

![Screenshot 2025-05-14 094400](https://github.com/user-attachments/assets/b33873bb-4321-40b9-a90f-d877c6e2267c)



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
