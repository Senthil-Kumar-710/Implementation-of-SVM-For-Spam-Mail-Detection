# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. Import the necessary packages.
2. Read the given csv file and display the few contents of the data.
3. Assign the features for x and y respectively.
4. Split the x and y sets into train and test sets.
5. Convert the Alphabetical data to numeric using CountVectorizer.
6. Predict the number of spam in the data using SVC (C-Support Vector Classification) method of SVM (Support vector machine) in sklearn library.
7. Find the accuracy of the model.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Senthil Kumar S
RegisterNumber:  212221230091
*/
import pandas as pd

data=pd.read_csv("/content/drive/MyDrive/Colab Notebooks/Semster 2/Intro to ML/spam.csv",encoding='latin-1')
data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:

## Dataset:
![1](https://user-images.githubusercontent.com/93860256/173338078-2795e420-3db2-4946-8fd6-eba69508a60d.PNG)


## Dataset information:
![2](https://user-images.githubusercontent.com/93860256/173338128-990ddbc9-d062-407a-889c-4541f52379a7.PNG)

![3](https://user-images.githubusercontent.com/93860256/173338177-b4ce78a3-2282-41c6-88ef-2891c751dddf.PNG)

## Detected spam:
![4](https://user-images.githubusercontent.com/93860256/173338269-7c63bab9-7e58-4f44-bba6-52f77e8bb392.PNG)



## Accuracy score of the model:
![5](https://user-images.githubusercontent.com/93860256/173338316-c8189eef-2339-4f65-880c-ce73f49321af.PNG)



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
