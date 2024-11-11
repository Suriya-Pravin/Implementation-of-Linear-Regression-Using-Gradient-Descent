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
import numpy as  np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(x1,y,learning_rate=0.01,num_iters=100):
  X=np.c_[np.ones(len(X1)),x1]
  theta=np.zeros(X.shape[1]).reshape(-1,1)

  for _ in range(num_iters):
    predictions=(X).dot(theta).reshape(-1,1)
    errors=(predictions-y).reshape(-1,1)        
    theta=learning=learning_rate*(1/len(X1))*X.T.dot(errors)
  return theta

data=pd.read_csv("50_Startups.csv")
print(data.head())
X=(data.iloc[1:,:-2].values)
print(X)
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
print(y)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X1_Scaled)
print(Y1_Scaled)
theta=linear_regression(X1_Scaled,Y1_Scaled);

new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted value: {pre}")

```

## Output:
### Data head:
![image](https://github.com/user-attachments/assets/c00a74e1-1a3c-4470-b76d-5ebd67a79337)

### X values:
![image](https://github.com/user-attachments/assets/1cf47201-f31c-47db-978e-cc2673dcb2fe)

### Y Values:
![image](https://github.com/user-attachments/assets/f70f3c10-f675-481a-b130-02adc5155162)

### X_Scaled:
![image](https://github.com/user-attachments/assets/1cdf0fad-e74e-4e41-8ccd-c815a2cda5f5)

### Y_Scaled:
![image](https://github.com/user-attachments/assets/d8b3b909-db3d-486a-b712-64e6db3ab09c)

### Predicted Value:
![image](https://github.com/user-attachments/assets/835bf422-224a-47de-ae59-c3949d20f36d)



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
