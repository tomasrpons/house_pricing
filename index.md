## Data description
The first thing to do would be to take a look at our data. In this particular case, data was provided in 2 different datasets, they contain similar information, but given that they have almost the same amount of rows but a few more variables, we decided to only use the one with the most amount of information.


```python
import pandas as pd
import numpy as np
```


```python
df = pd.read_csv('data/housing_clean_2.csv')
df = df.drop(columns=['Unnamed: 0'])
```

Now lets see the data


```python
df.head(5)
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
      <th>price</th>
      <th>space</th>
      <th>room</th>
      <th>bedroom</th>
      <th>furniture</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>city_area</th>
      <th>floor</th>
      <th>max_floor</th>
      <th>apartment_type</th>
      <th>renovation_type</th>
      <th>balcony</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>107100</td>
      <td>28.0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>41.724521</td>
      <td>44.753788</td>
      <td>Saburtalo District</td>
      <td>11</td>
      <td>11</td>
      <td>new</td>
      <td>newly renovated</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>257000</td>
      <td>72.0</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>41.689502</td>
      <td>44.820050</td>
      <td>Isani District</td>
      <td>15</td>
      <td>16</td>
      <td>new</td>
      <td>newly renovated</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>157200</td>
      <td>53.0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>41.677084</td>
      <td>44.817222</td>
      <td>Krtsanisi District</td>
      <td>2</td>
      <td>4</td>
      <td>new</td>
      <td>white frame</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>237200</td>
      <td>80.0</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>41.682883</td>
      <td>44.823815</td>
      <td>Krtsanisi District</td>
      <td>3</td>
      <td>4</td>
      <td>new</td>
      <td>white frame</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>158200</td>
      <td>60.0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>41.768762</td>
      <td>44.724123</td>
      <td>Saburtalo District</td>
      <td>14</td>
      <td>16</td>
      <td>old</td>
      <td>newly renovated</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 29204 entries, 0 to 29203
    Data columns (total 13 columns):
     #   Column           Non-Null Count  Dtype  
    ---  ------           --------------  -----  
     0   price            29204 non-null  int64  
     1   space            29204 non-null  float64
     2   room             29204 non-null  int64  
     3   bedroom          29204 non-null  int64  
     4   furniture        29204 non-null  int64  
     5   latitude         28958 non-null  float64
     6   longitude        28958 non-null  float64
     7   city_area        29204 non-null  object 
     8   floor            29204 non-null  int64  
     9   max_floor        29204 non-null  int64  
     10  apartment_type   29194 non-null  object 
     11  renovation_type  29204 non-null  object 
     12  balcony          29204 non-null  int64  
    dtypes: float64(3), int64(7), object(3)
    memory usage: 2.9+ MB
    


```python
df.describe()
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
      <th>price</th>
      <th>space</th>
      <th>room</th>
      <th>bedroom</th>
      <th>furniture</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>floor</th>
      <th>max_floor</th>
      <th>balcony</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2.920400e+04</td>
      <td>29204.000000</td>
      <td>29204.000000</td>
      <td>29204.000000</td>
      <td>29204.000000</td>
      <td>28958.000000</td>
      <td>28958.000000</td>
      <td>29204.000000</td>
      <td>29204.000000</td>
      <td>29204.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2.988408e+05</td>
      <td>87.569128</td>
      <td>2.923298</td>
      <td>1.819751</td>
      <td>0.430592</td>
      <td>41.729477</td>
      <td>44.781414</td>
      <td>6.361663</td>
      <td>10.744556</td>
      <td>0.776092</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.952151e+06</td>
      <td>45.717693</td>
      <td>1.050128</td>
      <td>0.852494</td>
      <td>0.495168</td>
      <td>0.051541</td>
      <td>0.083022</td>
      <td>4.707854</td>
      <td>5.454033</td>
      <td>0.416868</td>
    </tr>
    <tr>
      <th>min</th>
      <td>3.300000e+03</td>
      <td>13.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>40.469229</td>
      <td>38.829869</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.483000e+05</td>
      <td>57.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>41.708101</td>
      <td>44.753788</td>
      <td>3.000000</td>
      <td>7.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2.142000e+05</td>
      <td>75.000000</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>41.724521</td>
      <td>44.772078</td>
      <td>5.000000</td>
      <td>10.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3.394000e+05</td>
      <td>105.000000</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>41.734181</td>
      <td>44.810649</td>
      <td>9.000000</td>
      <td>14.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>3.295100e+08</td>
      <td>530.000000</td>
      <td>14.000000</td>
      <td>4.000000</td>
      <td>1.000000</td>
      <td>44.266066</td>
      <td>47.238277</td>
      <td>127.000000</td>
      <td>113.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



It seems like we don't have clear outliers, so we will leave them because they might represent an outstanding (or very poor) location. Now lets plot the correlation matrix to see if we have some clear correlation between variables.

## Data Visualization


```python
# Definition of a function to visualize correlation between variables
import seaborn as sn
import matplotlib.pyplot as plt


def plot_correlation(df):
    corr_matrix = df.corr()
    heat_map = sn.heatmap(corr_matrix, annot=False)
    plt.show(heat_map)
```


```python
plot_correlation(df)
```


![png](output_11_0.png)


Given that our target variable is price, we cannot see a clear correlation. Lets proceed divide the data into train and test set. Using the following graph we can see that we don't have clear outliers in our dataset.


```python
pd.plotting.scatter_matrix(df, alpha=0.2, figsize=(10, 10));
```


![png](output_13_0.png)


## Data preprocessing
We will proceed to separate data into train and test. Once we have our data ready we will begin imputing and normalizing/standardizing.


```python
from sklearn.model_selection import train_test_split

X = df.drop(axis=1, columns='price')
# We are using a log to reduce the scale of the prices.
y = np.log(df['price'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=123)
```

Now we will separate numerical and categorical variables


```python
# Numerical variable names
num_var = X_train.select_dtypes(np.number).columns
# Categorical variable names
cat_var = X_train.select_dtypes(include=['object', 'bool']).columns
```

### Data Cleaning
This is the time to take care of missing values, we will use KNN-Imputer to deal with numerical missing values and 'most frequent' simple imputer to deal with categorical ones


```python
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.compose import ColumnTransformer
```


```python
# Creating both numerical and categorical imputer
t1 = ('num_imputer', KNNImputer(n_neighbors=5), num_var)
t2 = ('cat_imputer', SimpleImputer(strategy='most_frequent'),
      cat_var)

column_transformer_cleaning = ColumnTransformer(
    transformers=[t1, t2], remainder='passthrough')

column_transformer_cleaning.fit(X_train)

Train_transformed = column_transformer_cleaning.transform(X_train)
Test_transformed = column_transformer_cleaning.transform(X_test)

# Here we update the order in wich variables are located in the dataframe, given that after transforming, we will have all
# numerical variables first, followed by all the categorical variables.
var_order = num_var.tolist() + cat_var.tolist()

# And finally we recreate the Data Frames
X_train_clean = pd.DataFrame(Train_transformed, columns=var_order)
X_test_clean = pd.DataFrame(Test_transformed, columns=var_order)
```

### Normalizing and Enconding data
Next step is to normalize and enconde data to achieve better performance in our models


```python
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
```


```python
# We obtain the diferent values in all categorical variables
dif_values = [df[column].dropna().unique() for column in cat_var]
```


```python
# Now we create the transformers
t_norm = ("normalizer", MinMaxScaler(feature_range=(0, 1)), num_var)
t_nominal = ("onehot", OneHotEncoder(
    sparse=False, categories=dif_values), cat_var)
# As the dataset isn't huge, we will set sparse=false
```


```python
column_transformer_norm_enc = ColumnTransformer(transformers=[t_norm, t_nominal],
                                                remainder='passthrough')

column_transformer_norm_enc.fit(X_train_clean);
```


```python
X_train_transformed = column_transformer_norm_enc.transform(X_train_clean)
X_test_transformed = column_transformer_norm_enc.transform(X_test_clean)
```

## Model Selection
We will create pipelines to fine tune hyperparameters using Grid Search. This step will take place in the train set to avoid having any contact with the test set (obviously).


```python
from sklearn.metrics import mean_squared_error
def root_mean_squared_error(y_true, y_pred):
    ''' Root mean squared error regression loss
    
    Parameters
    ----------
    y_true : array-like of shape = (n_samples) or (n_samples, n_outputs)
    Ground truth (correct) target values.

    y_pred : array-like of shape = (n_samples) or (n_samples, n_outputs)
    Estimated target values.
    '''
    return np.sqrt(mean_squared_error(y_true, y_pred, squared=True))
```


```python
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_log_error, mean_squared_error

def reg_report(model, X, y):
    print(f"R2 score : {r2_score(y,model.predict(X)):.2f}")
    print(f"MAE loss : {mean_absolute_error(y,model.predict(X)):.2f}")
    print(f"RMSE loss : {np.sqrt(mean_squared_error(y,model.predict(X))):.2f}")
```


```python
from sklearn.model_selection import validation_curve
from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer
```

### Grid search for Decision Tree Regressor


```python
from sklearn import tree

rmse_scorer = make_scorer(root_mean_squared_error, greater_is_better=False)
pipe_tree = make_pipeline(tree.DecisionTreeRegressor(random_state=1))

# make an array of depths to choose from, say 1 to 20
depths = np.arange(1, 21)
num_leafs = [1, 5, 10, 20, 50, 100]

param_grid_tree = [{'decisiontreeregressor__max_depth': depths,
                    'decisiontreeregressor__min_samples_leaf': num_leafs}]
```


```python
from sklearn.model_selection import GridSearchCV

gs_tree = GridSearchCV(estimator=pipe_tree,
                       param_grid=param_grid_tree, scoring=rmse_scorer, cv=10)
best_tree = gs_tree.fit(X_train_transformed, y_train)
```


```python
#print(reg_report(best_tree, X_train_transformed, y_train))
#print('Best attributes: ', best_tree.best_params_)
```

Remember MAE and RMSE are compared to the logarithm of the price (**that has mean=12.34**)

R2 score : 0.87  
MAE loss : 0.15  
RMSE loss : 0.23  
  
Best attributes:  {'decisiontreeregressor__max_depth': 15, 'decisiontreeregressor__min_samples_leaf': 20}  

### Grid search for Linear Regression


```python
from sklearn.linear_model import LinearRegression

pipe_lr = make_pipeline(LinearRegression())
fit_intercept = [True, False]
normalize = [True, False]
copy_x = [True, False]

parameters = [{'linearregression__fit_intercept': fit_intercept,
               'linearregression__normalize': normalize, 'linearregression__copy_X': copy_x}]

gs_lr = GridSearchCV(estimator=pipe_lr, param_grid=parameters,
                     scoring=rmse_scorer, cv=10)
best_lr=gs_lr.fit(X_train_transformed, y_train)

```


```python
#print(reg_report(best_lr, X_train_transformed, y_train))
#print('Best attributes: ', best_tree.best_params_)
```

R2 score : 0.77  
MAE loss : 0.21  
RMSE loss : 0.30  
  
Best attributes:  {'decisiontreeregressor__max_depth': 15, 'decisiontreeregressor__min_samples_leaf': 20}

## Model Training
Both models performed well, but our winner is **Decision Tree Classifier**. Now it's time to train that model with all of our train data to obtain the *down to earth* performance of our model.


```python
model = tree.DecisionTreeRegressor(random_state=1, max_depth=15,min_samples_leaf=20)
model.fit(X_train_transformed,y_train)

y_pred = model.predict(X_test_transformed)

# We use the exponencial function to show the real RMSE of the price
RMSE = np.exp(np.sqrt(mean_squared_error(y_test, y_pred, squared=True)))
```


```python
#print('Real World RMSE: ',round(RMSE,4))
```

Real World RMSE:  1.2905

## Conclusion
To conclude, we created this tool to predict house pricing based on certain parameters. We achieved a RMSE of 1.29 U$D
