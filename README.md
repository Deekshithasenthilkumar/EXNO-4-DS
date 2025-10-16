# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```python
import pandas as pd
import numpy as np
df=pd.read_csv("bmi.csv")
df
```
<img width="479" height="527" alt="image" src="https://github.com/user-attachments/assets/a8ba3752-ab16-4c24-b45c-6cb81652c797" />

```python
df.head()
```
<img width="442" height="263" alt="image" src="https://github.com/user-attachments/assets/0ec9efe3-b85d-460e-aac5-45813cf68081" />

```python
df.dropna()
```
<img width="519" height="522" alt="image" src="https://github.com/user-attachments/assets/96dbbbd0-6b3c-48d0-abf7-df7128dd5e12" />

```python
max_vals=np.max(np.abs(df[['Height','Weight']]))
max_vals
```
<img width="215" height="65" alt="image" src="https://github.com/user-attachments/assets/f5974ae6-98c8-431a-86e2-62445985334d" />

```python
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```
<img width="456" height="455" alt="image" src="https://github.com/user-attachments/assets/20d71c58-d7ea-49f9-b9b3-4572006799dd" />

```python
df1=pd.read_csv("bmi.csv")
```
```python
df2=pd.read_csv("bmi.csv")
```
```python
df3=pd.read_csv("bmi.csv")
```
```python
df4=pd.read_csv("bmi.csv")
```
```python
df5=pd.read_csv("bmi.csv")
```
```python
df1
```
<img width="436" height="516" alt="image" src="https://github.com/user-attachments/assets/000ce304-679f-45f0-a472-61bbbc27f14b" />

```python
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df1[['Height','Weight']]=sc.fit_transform(df1[['Height','Weight']])
df.head(10)
```
<img width="478" height="434" alt="image" src="https://github.com/user-attachments/assets/fdabc51a-ce30-4526-9fcf-038725845ba7" />

```python
from sklearn.preprocessing import MaxAbsScaler
max1=MaxAbsScaler()
df3[['Height','Weight']]=max1.fit_transform(df3[['Height','Weight']])
df3
```
<img width="444" height="521" alt="image" src="https://github.com/user-attachments/assets/1a4cad27-fb84-4dd0-84df-efecfac60320" />

```python
from sklearn.preprocessing import RobustScaler
roub=RobustScaler()
df4[['Height','Weight']]=roub.fit_transform(df4[['Height','Weight']])
df4
```
<img width="449" height="514" alt="image" src="https://github.com/user-attachments/assets/971f84ed-e7a5-494d-bee3-7e70417e0803" />

```python
from sklearn.feature_selection import SelectKBest,f_regression,mutual_info_classif
from sklearn.feature_selection import chi2
data=pd.read_csv("income(1) (1).csv")
data
```
<img width="1826" height="530" alt="image" src="https://github.com/user-attachments/assets/81272d44-6c9b-4f19-baee-ae7d17b3d4f3" />

```python
data1=pd.read_csv('/content/titanic_dataset (1).csv')
data1
```
<img width="1703" height="665" alt="image" src="https://github.com/user-attachments/assets/91852a46-10f6-4d38-85fa-99755e8ee865" />

```python
data1=data1.dropna()
x=data1.drop(['Survived','Name','Ticket'],axis=1)
y=data1['Survived']
data1['Sex']=data1['Sex'].astype('category')
data1['Cabin']=data1['Cabin'].astype('category')
data1['Embarked']=data1['Embarked'].astype('category')
```
```python
data1['Sex']=data1['Sex'].cat.codes
data1['Cabin']=data1['Cabin'].cat.codes
data1['Embarked']=data1['Embarked'].cat.codes
```
```python
data1
```
<img width="1597" height="627" alt="image" src="https://github.com/user-attachments/assets/86e84c09-5f0f-49a5-a12c-c85713cb1c0e" />

```python
k=5
selector=SelectKBest(score_func=chi2,k=k)
x=pd.get_dummies(x)
x_new=selector.fit_transform(x,y)
```
```python
x_encoded=pd.get_dummies(x)
selector=SelectKBest(score_func=chi2,k=5)
x_new=selector.fit_transform(x_encoded,y)
```
```python
selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
<img width="965" height="85" alt="image" src="https://github.com/user-attachments/assets/b42a45ea-4765-40ea-b519-87566f79baf7" />

```python
selector=SelectKBest(score_func=f_regression,k=5)
x_new=selector.fit_transform(x_encoded,y)
selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
<img width="1025" height="85" alt="image" src="https://github.com/user-attachments/assets/38010ad6-f879-4ba9-b0fd-e346b1fa8e97" />

```python
selector=SelectKBest(score_func=mutual_info_classif,k=5)
x_new=selector.fit_transform(x,y)
selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
<img width="1003" height="93" alt="image" src="https://github.com/user-attachments/assets/751b304a-8ac0-494d-8456-c9cf01c814d8" />

```python
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
```
```python
model=RandomForestClassifier()
sfm=SelectFromModel(model,threshold='mean')
x=pd.get_dummies(x)
sfm.fit(x,y)
selected_features=x.columns[sfm.get_support()]
print("Selected Features:")
print(selected_features)
```
<img width="929" height="147" alt="image" src="https://github.com/user-attachments/assets/6a3635f0-fd40-40c8-b3fe-ecf99cbb2f37" />

```python
from sklearn.ensemble import RandomForestClassifier
```
```python
model=RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(x,y)
feature_selection=model.feature_importances_
threshold=0.1
selected_features=x.columns[feature_selection>threshold]
print("Selected Features:")
print(selected_features)
```
<img width="961" height="101" alt="image" src="https://github.com/user-attachments/assets/c43ce1c5-a370-4478-b347-998f18139650" />

```python
model=RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(x,y)
feature_importance=model.feature_importances_
threshold=0.15
selected_features=x.columns[feature_importance>threshold]
print("Selected Features:")
print(selected_features)
```
<img width="359" height="65" alt="image" src="https://github.com/user-attachments/assets/cdecf840-5603-4657-b5f5-146e322a07d0" />


# RESULT:
Thus the feature selection and feature scaling has been used on the given dataset and executed successfully.
