# CS334FinalProject House Sale Price Prediction

Instructions To Use Prediction App:

1. git clone https://github.com/qyccc3/CS334FinalProject.git
2. cd CS334FinalProject
3. python HousePricePredictionApp.py

**Make sure you have python3 installed and tkinter, pandas, sklearn library installed to use the application.**


# Data Processing and Models Used

Dataset: Ames Housing Dataset collecting from [https://www.kaggle.com/datasets/prevek18/ames-housing-dataset]()

We've manually selected and deleted the dataset from a total of 82 attributes to 31 attributes at the beginning.


```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
dataset = pd.read_csv("./ManualPreprocessedAmesHousing.csv")
dataset.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MS SubClass</th>
      <th>MS Zoning</th>
      <th>Lot Frontage</th>
      <th>Lot Area</th>
      <th>Lot Shape</th>
      <th>Lot Config</th>
      <th>Bldg Type</th>
      <th>House Style</th>
      <th>Overall Qual</th>
      <th>Overall Cond</th>
      <th>...</th>
      <th>Paved Drive</th>
      <th>Wood Deck SF</th>
      <th>Open Porch SF</th>
      <th>Enclosed Porch</th>
      <th>3Ssn Porch</th>
      <th>Screen Porch</th>
      <th>Pool Area</th>
      <th>Sale Type</th>
      <th>Sale Condition</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>20</td>
      <td>RL</td>
      <td>141</td>
      <td>31770</td>
      <td>IR1</td>
      <td>Corner</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>6</td>
      <td>5</td>
      <td>...</td>
      <td>P</td>
      <td>210</td>
      <td>62</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>WD</td>
      <td>Normal</td>
      <td>215.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20</td>
      <td>RH</td>
      <td>80</td>
      <td>11622</td>
      <td>Reg</td>
      <td>Inside</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>5</td>
      <td>6</td>
      <td>...</td>
      <td>Y</td>
      <td>140</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>120</td>
      <td>0</td>
      <td>WD</td>
      <td>Normal</td>
      <td>105.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20</td>
      <td>RL</td>
      <td>81</td>
      <td>14267</td>
      <td>IR1</td>
      <td>Corner</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>6</td>
      <td>6</td>
      <td>...</td>
      <td>Y</td>
      <td>393</td>
      <td>36</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>WD</td>
      <td>Normal</td>
      <td>172.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20</td>
      <td>RL</td>
      <td>93</td>
      <td>11160</td>
      <td>Reg</td>
      <td>Corner</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>7</td>
      <td>5</td>
      <td>...</td>
      <td>Y</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>WD</td>
      <td>Normal</td>
      <td>244.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>60</td>
      <td>RL</td>
      <td>74</td>
      <td>13830</td>
      <td>IR1</td>
      <td>Inside</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>5</td>
      <td>5</td>
      <td>...</td>
      <td>Y</td>
      <td>212</td>
      <td>34</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>WD</td>
      <td>Normal</td>
      <td>189.9</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 55 columns</p>
</div>




```python
plt.figure(figsize=(30, 30))
sns.heatmap(dataset.corr())
plt.show()
```

    /var/folders/kx/dd0tmbz95_n461n7xxtqy1wm0000gn/T/ipykernel_36849/2353200527.py:2: FutureWarning: The default value of numeric_only in DataFrame.corr is deprecated. In a future version, it will default to False. Select only valid columns or specify the value of numeric_only to silence this warning.
      sns.heatmap(dataset.corr())



    
![png](README_files/README_2_1.png)
    



```python
dataset.hist(figsize=(30, 30))
plt.show()
```


    
![png](README_files/README_3_0.png)
    



```python
for column in dataset.columns:
    if column in ['Gr Liv Area', '1st Flr SF', 'Garage Area', 'Overall Qual', 'Total Bsmt SF']:
        fig, ax = plt.subplots(figsize=(5,5))
        ax.scatter(dataset['SalePrice'], dataset[column])
        ax.plot
        ax.set_xlabel('SalePrice')
        ax.set_ylabel(column)
        plt.show()
# Only Show The Top 5 Correlated Features
```


    
![png](README_files/README_4_0.png)
    



    
![png](README_files/README_4_1.png)
    



    
![png](README_files/README_4_2.png)
    



    
![png](README_files/README_4_3.png)
    



    
![png](README_files/README_4_4.png)
    


After reviewing the scatter plots bewteen SalePrice and other columns, we picked the 5 most deteminant attributes that show high correlation with SalePrice: '**Gr Liv Area**', '**1st Flr SF**', '**Garage Area**', '**Overall Qual**', '**Total Bsmt SF**', '**SalePrice**' 

# Models Used
**Decision Tree**\
**Linear Regression**\
**KMeans**

We categorized SalePrice for classification models based on 4 quantiles of the dataset:


```python
house_data = pd.read_csv("./ManualPreprocessedAmesHousing.csv")[['Gr Liv Area', '1st Flr SF', 'Garage Area', 'Overall Qual', 'Total Bsmt SF','SalePrice']]
print("25 percentile of the data")
print('\t',int(house_data['SalePrice'].quantile(0.25)*1000), '== 0')
print("50 percentile of the data")
print('\t',int(house_data['SalePrice'].quantile(0.5)*1000), '== 1')
print("75 percentile of the data")
print('\t',int(house_data['SalePrice'].quantile(0.75)*1000), '== 2')
print("90 percentile of the data")
print('\t',int(house_data['SalePrice'].quantile(0.9)*1000), '== 3')
```

    25 percentile of the data
    	 129500 == 0
    50 percentile of the data
    	 160000 == 1
    75 percentile of the data
    	 213500 == 2
    90 percentile of the data
    	 281241 == 3



```python
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn import tree
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
import graphviz
dataset = pd.read_csv("./ManualPreprocessedAmesHousingClassification.csv")[['Gr Liv Area', '1st Flr SF', 'Garage Area', 'Overall Qual', 'Total Bsmt SF','SalePrice']]
```

## Decision Tree


```python
from sklearn.model_selection import KFold
kf = KFold(n_splits=10, random_state=100, shuffle=True)
kf.get_n_splits(dataset)
depth = range(1,20)
accuracy = []
for max_d in depth:
    avg_accuracy = 0
    for train, test in kf.split(dataset):
        train_data = dataset.iloc[train]
        test_data = dataset.iloc[test]
        dt = DecisionTreeClassifier(criterion='gini', max_depth=max_d, random_state=100)
        dt.fit(train_data.drop('SalePrice', axis=1), train_data['SalePrice'])
        y_pred = dt.predict(test_data.drop('SalePrice', axis=1))
        # find average
        avg_accuracy += accuracy_score(test_data['SalePrice'], y_pred)
    accuracy.append(avg_accuracy/10*100)
plt.plot(depth, accuracy)
plt.xticks(depth)
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Max Depth using Gini Index')
plt.show()
```


    
![png](README_files/README_11_0.png)
    



```python
avg_accuracy = 0
print("10 Fold Cross Validation Using Gini Index, Max Depth 6: ")
for train, test in kf.split(dataset):
    train_data = dataset.iloc[train]
    test_data = dataset.iloc[test]
    dt = DecisionTreeClassifier(criterion='gini', max_depth=6, random_state=100)
    dt.fit(train_data.drop('SalePrice', axis=1), train_data['SalePrice'])
    y_pred = dt.predict(test_data.drop('SalePrice', axis=1))
    # find average
    avg_accuracy += accuracy_score(test_data['SalePrice'], y_pred)
    # print("\tAccuracy is ", accuracy_score(test_data['SalePrice'], y_pred)*100)
print("Average accuracy is ", avg_accuracy/10*100)
avg_accuracy = 0
print("10 Fold Cross Validation Using Entropy, Max Depth 6:")
for train, test in kf.split(dataset):
    train_data = dataset.iloc[train]
    test_data = dataset.iloc[test]
    dt = DecisionTreeClassifier(criterion='entropy', max_depth=6, random_state=100)
    dt.fit(train_data.drop('SalePrice', axis=1), train_data['SalePrice'])
    y_pred = dt.predict(test_data.drop('SalePrice', axis=1))
    # find average
    avg_accuracy += accuracy_score(test_data['SalePrice'], y_pred)
    # print("\tAccuracy is ", accuracy_score(test_data['SalePrice'], y_pred)*100)
print("Average accuracy is ", avg_accuracy/10*100)
```

    10 Fold Cross Validation Using Gini Index, Max Depth 6: 
    Average accuracy is  67.98634812286689
    10 Fold Cross Validation Using Entropy, Max Depth 6:
    Average accuracy is  66.9283276450512



```python
# Best Max Depth is 6, using Gini Index
dt = DecisionTreeClassifier(criterion='gini', max_depth=6, random_state=100)
dt.fit(train_data.drop('SalePrice', axis=1), train_data['SalePrice'])
y_pred = dt.predict(test_data.drop('SalePrice', axis=1))
print("Accuracy is ", accuracy_score(test_data['SalePrice'], y_pred)*100)
```

    Accuracy is  67.23549488054607



```python
class_names = ['0', '1', '2', '3']
dot_data = tree.export_graphviz(dt, out_file=None,
                                feature_names = train_data.drop('SalePrice', axis=1).columns,
                                class_names = class_names,
                                filled = True, rounded = True,
                                special_characters = True)
graph = graphviz.Source(dot_data) 
graph
```




    
![svg](README_files/README_14_0.svg)
    



## KMeans


```python
"""
Fit KMeans to the data without applying PCA
"""
kf = KFold(n_splits=10)
kf.get_n_splits(dataset)

avg_mse = 0
avg_acc = 0
for train, test in kf.split(dataset):
    train_data = dataset.iloc[train]
    test_data = dataset.iloc[test]
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(train_data.drop("SalePrice", axis=1),train_data['SalePrice'])
    y_pred = kmeans.predict(test_data.drop("SalePrice", axis=1))
    # print("Mean Squared Error: ", mean_squared_error(y_pred, test_data['SalePrice']))
    # print("Arrucary: ", accuracy_score(y_pred, test_data['SalePrice']))
    avg_mse += mean_squared_error(y_pred, test_data['SalePrice'])
    avg_acc += accuracy_score(y_pred, test_data['SalePrice'])
print("Average Mean Squared Error: ", avg_mse/10)
print("Average Arrucary: ", avg_acc/10)
```

    Average Mean Squared Error:  2.58122866894198
    Average Arrucary:  0.22218430034129694



```python
"""
Fit KMeans to the data after applying PCA
"""
pca = PCA(n_components=2)
pca.fit(dataset.drop('SalePrice', axis=1))
x_data = pca.transform(dataset.drop('SalePrice', axis=1))
x_data = pd.DataFrame(x_data)
y_data = dataset['SalePrice']
y_data = pd.DataFrame(y_data)
kf = KFold(n_splits=10)
kf.get_n_splits(dataset)
avg_mse = 0
avg_acc = 0
for train, test in kf.split(x_data):
    x_train = x_data.iloc[train]
    x_test = x_data.iloc[test]
    y_train = y_data.iloc[train]
    y_test = y_data.iloc[test]
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(x_train,y_train)
    y_pred = kmeans.predict(x_test)
    # print("Mean Squared Error: ", mean_squared_error(y_pred, test_data['SalePrice']))
    # print("Arrucary: ", accuracy_score(y_pred, test_data['SalePrice']))
    avg_mse += mean_squared_error(y_pred, y_test)
    avg_acc += accuracy_score(y_pred, test_data['SalePrice'])
print("Average Mean Squared Error: ", avg_mse/10)
print("Average Arrucary: ", avg_acc/10)
```

    Average Mean Squared Error:  2.1901023890784983
    Average Arrucary:  0.2563139931740614


## Linear Regression


```python
# Linear Regression on Original Data
kFold = KFold(n_splits=10, shuffle=True, random_state=0)
dataset_orig = pd.get_dummies(pd.read_csv("./ManualPreprocessedAmesHousing.csv"))
X_orig = dataset_orig.drop(columns=["SalePrice"])
y_orig = dataset_orig["SalePrice"]
kFold = KFold(n_splits=10, shuffle=True, random_state=0)
avgMSE = 0
avgR2 = 0
for train, test in kFold.split(X_orig):
    X_train_fold, X_test_fold = X_orig.iloc[train], X_orig.iloc[test]
    y_train_fold, y_test_fold = y_orig.iloc[train], y_orig.iloc[test]
    model = LinearRegression()
    model.fit(X_train_fold, y_train_fold)
    y_pred = model.predict(X_test_fold)
    avgMSE += mean_squared_error(y_test_fold, y_pred)
    avgR2 += model.score(X_test_fold, y_test_fold)
print("Average MSE: ", avgMSE/10)
print("Average R2: ", avgR2/10)
```

    Average MSE:  935.6220729544648
    Average R2:  0.8555372323665559



```python
"""
Linear Regression on Filtered Data
"""
dataset_filtered = pd.read_csv("./ManualPreprocessedAmesHousing.csv")[['Gr Liv Area', '1st Flr SF', 'Garage Area', 'Overall Qual', 'Total Bsmt SF','SalePrice']]
X_filtered = dataset_filtered.drop(columns=["SalePrice"])
y_filtered = dataset_filtered["SalePrice"]
kFold = KFold(n_splits=10, shuffle=True, random_state=0)
avgMSE = 0
avgR2 = 0
for train, test in kFold.split(X_filtered):
    X_train_fold, X_test_fold = X_filtered.iloc[train], X_filtered.iloc[test]
    y_train_fold, y_test_fold = y_filtered.iloc[train], y_filtered.iloc[test]
    model = LinearRegression()
    model.fit(X_train_fold, y_train_fold)
    y_pred = model.predict(X_test_fold)
    avgMSE += mean_squared_error(y_test_fold, y_pred)
    avgR2 += model.score(X_test_fold, y_test_fold)
print("Average MSE: ", avgMSE/10)
print("Average R2: ", avgR2/10)
```

    Average MSE:  1428.9737943338919
    Average R2:  0.7777471605626946



```python
lr = LinearRegression()
lr.fit(dataset_filtered.drop('SalePrice', axis=1), dataset_filtered['SalePrice'])
y_pred = lr.predict(dataset_filtered.drop('SalePrice', axis=1))
print("Mean Squared Error: ", mean_squared_error(y_pred, dataset_filtered['SalePrice']))
print("R2: ", model.score(dataset_filtered.drop('SalePrice', axis=1), dataset_filtered['SalePrice']))
fig, ax = plt.subplots(figsize=(7, 7))
plt.scatter(dataset_filtered['SalePrice'], y_pred)
x = np.linspace(0, 600, 1000)
plt.plot(x, x, color='red')
plt.xlabel("Actual Sale Price")
plt.ylabel("Predicted Sale Price")
plt.show()
```

    Mean Squared Error:  1400.2244916039772
    R2:  0.7803537073215394



    
![png](README_files/README_21_1.png)
    



```python

```
