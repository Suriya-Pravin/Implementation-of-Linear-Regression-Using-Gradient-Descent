### Date: 
# Ex-3:Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1)Import necessary libraries such as NumPy, Pandas, Matplotlib, and metrics from sklearn.</br>
2)Load the dataset into a Pandas DataFrame and preview it using head() and tail().</br>
3)Extract the independent variable X and dependent variable Y from the dataset.</br>
4)Initialize the slope m and intercept c to zero. Set the learning rate L and define the number of epochs.</br>
5)Plot the error against the number of epochs to visualize the convergence.</br>
6)Display the final values of m and c, and the error plot.</br>

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: 212223230223
RegisterNumber:  Suriya Pravin M
*/
import numpy as np
import pandas as pd
from sklearn.metrics import  mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt

dataset = pd.read_csv('student_scores.csv')
print(dataset.head())
print(dataset.tail())

dataset.info()

X=dataset.iloc[:,:-1].values
print(X)
Y=dataset.iloc[:,-1].values
print(Y)

print(X.shape)
print(Y.shape)

m=0
c=0
L=0.0001
epochs=5000
n=float(len(X))
error=[]
for i in range(epochs):
    Y_pred = m*X +c
    D_m=(-2/n)*sum(X *(Y-Y_pred))
    D_c=(-2/n)*sum(Y -Y_pred)
    m=m-L*D_m
    c=c-L*D_c
    error.append(sum(Y-Y_pred)**2)
print(m,c)
type(error)
print(len(error))

plt.plot(range(0,epochs),error)

```

## Output:
![1](https://github.com/user-attachments/assets/e57d3087-7f62-4857-b738-ade3ea123b2d)
![2](https://github.com/user-attachments/assets/d8db4f9a-ef47-4f72-b701-9be18ff1c245)
![3](https://github.com/user-attachments/assets/a1cbe125-905d-4cf3-bc56-48b068d62cba)
![4](https://github.com/user-attachments/assets/395f7d39-b1e4-4344-9624-7f88ca2cbdcf)
![5](https://github.com/user-attachments/assets/35f04d1d-2a45-44b1-a2a0-c2c6b13e3d63)
![6](https://github.com/user-attachments/assets/e3149a7b-aa03-459d-89b1-ba7d90bb343f)
![7](https://github.com/user-attachments/assets/47f04b32-ea2f-48a1-b1a2-1069c93e7d3c)



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
