# House Pricing
Houses are expensive. For an average buyer it is quite hard to decide which house to buy. While online house retail sites somewhat alleviate that problem, the sheer amount of data that is present in them is daunting for any user. Wouldn't it be nice if there was an automated software that could predict a fair price for a house, given it's data? This is exactly why we developed this tool for.


## Data description
The first thing to do would be to take a look at our data. In this particular case, data was provided in 2 different datasets, they contain similar information, but given that they have almost the same amount of rows but a few more variables, we decided to only use the one with the most amount of information.


```python
import pandas as pd
import numpy as np
```


```python
df = pd.read_csv('data/data_from_json.csv')
df = df.drop(columns=['Unnamed: 0'])
```

Now lets see the data


```python
df.tail(5)
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
      <th>29199</th>
      <td>179200</td>
      <td>75.00</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>41.761308</td>
      <td>44.789730</td>
      <td>Nadzaladevi District</td>
      <td>11</td>
      <td>22</td>
      <td>construction</td>
      <td>green frame</td>
      <td>1</td>
    </tr>
    <tr>
      <th>29200</th>
      <td>126600</td>
      <td>53.00</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>41.731884</td>
      <td>44.836876</td>
      <td>Other</td>
      <td>8</td>
      <td>12</td>
      <td>new</td>
      <td>green frame</td>
      <td>1</td>
    </tr>
    <tr>
      <th>29201</th>
      <td>62400</td>
      <td>25.75</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>41.731884</td>
      <td>44.836876</td>
      <td>Other</td>
      <td>7</td>
      <td>12</td>
      <td>new</td>
      <td>green frame</td>
      <td>1</td>
    </tr>
    <tr>
      <th>29202</th>
      <td>167200</td>
      <td>70.00</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>41.731884</td>
      <td>44.836876</td>
      <td>Other</td>
      <td>4</td>
      <td>12</td>
      <td>new</td>
      <td>green frame</td>
      <td>1</td>
    </tr>
    <tr>
      <th>29203</th>
      <td>169300</td>
      <td>75.00</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>41.731884</td>
      <td>44.836876</td>
      <td>Other</td>
      <td>5</td>
      <td>8</td>
      <td>construction</td>
      <td>green frame</td>
      <td>1</td>
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

pd.plotting.scatter_matrix(df, alpha=0.2, figsize=(10, 10));

## Data preprocessing
We will proceed to separate data into train and test. Once we have our data ready we will begin imputing and normalizing/standardizing.


```python
from sklearn.model_selection import train_test_split

X = df.drop(axis=1, columns='price')
y = df['price']

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

### Unsupervised learning


```python
from sklearn.cluster import KMeans
```


```python
kmeans = KMeans(
    n_clusters=2, 
    random_state=123
).fit(X_train_transformed)
```


```python
test_cluster = kmeans.predict(X_test_transformed)
```


```python
X_train_transformed = np.append(X_train_transformed, np.expand_dims(kmeans.labels_, axis=1), axis=1)
X_test_transformed = np.append(X_test_transformed, np.expand_dims(test_cluster, axis=1), axis=1)
```

### Feature Engineering


```python
X_val_train, X_val_test, y_val_train, y_val_test = train_test_split(
    X_train_transformed, y_train, test_size=0.30, random_state=123)
```


```python
from sklearn.ensemble import ExtraTreesRegressor
import matplotlib.pyplot as plt
model = ExtraTreesRegressor()
model.fit(X_val_train, y_val_train)
print(model.feature_importances_)
```

    [1.88279024e-01 5.31595550e-02 6.62028485e-02 1.60783376e-02
     2.56854814e-01 2.30141022e-01 8.62513302e-02 6.39541243e-02
     2.50864007e-02 1.30267890e-03 1.67187181e-05 4.46828279e-05
     1.57882755e-04 8.16312705e-06 1.58674287e-05 4.82876694e-04
     2.69620735e-03 1.13453585e-05 9.23340485e-06 1.94127613e-05
     9.37150900e-05 7.44937721e-05 1.75189974e-03 1.60082869e-04
     6.19480330e-04 2.89415264e-04 1.73421107e-05 5.24741647e-03
     4.61917562e-06 4.30801101e-05 9.25929859e-04]
    


```python
feat_importances = pd.Series(model.feature_importances_)
feat_importances.nlargest(30).plot(kind='barh', figsize=(10, 10))
plt.show()
```


![png](output_35_0.png)



```python
X_train_transformed = X_train_transformed[:, [4, 5, 0, 7, 2, 6]]
```


```python
X_test_transformed = X_test_transformed[:, [4, 5, 0, 7, 2, 6]]
```

### Outliers


```python
df_outlier = pd.DataFrame(data=X_train_transformed)
df_outlier = df_outlier.merge(y_train, on=df_outlier.index, indicator = False).drop(axis=1, columns='key_0')

df_outlier_test = pd.DataFrame(data=X_test_transformed)
df_outlier_test = df_outlier_test.merge(y_test, on=df_outlier_test.index, indicator = False).drop(axis=1, columns='key_0')
```


```python
df_outlier = df_outlier.rename(columns={0: "space"})
```


```python
plt.figure(figsize=(10,5))
sn.scatterplot(x='space', y='price', 
                data=df_outlier, alpha=.6)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x19dd8f85670>




![png](output_41_1.png)



```python
from sklearn.ensemble import IsolationForest
```


```python
iso_forest = IsolationForest(contamination=0.5)
iso_forest.fit(df_outlier)
```




    IsolationForest(contamination=0.5)




```python
plt.figure(figsize=(10,5))
sn.scatterplot(x='space', y='price',
                data=df_outlier[iso_forest.predict(df_outlier) == 1], alpha=.6)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x19dd9bd26a0>




![png](output_44_1.png)



```python
df_outlier = df_outlier[iso_forest.predict(df_outlier) == 1]
X_train_transformed = df_outlier.drop(axis=1, columns='price')
y_train = df_outlier['price']
```


```python
df_outlier_test = df_outlier_test[iso_forest.predict(df_outlier_test) == 1]
X_test_transformed = df_outlier_test.drop(axis=1, columns='price')
y_test = df_outlier_test['price']
```

## Model Selection
We will create pipelines to fine tune hyperparameters using Grid Search. This step will take place in the train set to avoid having any contact with the test set (obviously).


```python
from sklearn.model_selection import validation_curve
from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from optuna.samplers import TPESampler
import optuna

from sklearn import tree
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from lightgbm import LGBMRegressor

from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
```


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
    print(f"error % : {np.sqrt(mean_squared_error(y,model.predict(X)))/np.mean(y):.2f}")
```


```python
X_val_train, X_val_test, y_val_train, y_val_test = train_test_split(
    X_train_transformed, y_train, test_size=0.20, random_state=123)
```

### Grid search for Decision Tree Regressor


```python
rmse_scorer = make_scorer(root_mean_squared_error, greater_is_better=False)
pipe_tree = make_pipeline(tree.DecisionTreeRegressor(random_state=1))

depths = np.arange(1, 21)
num_leafs = [1, 5, 10, 20, 50, 100]

param_grid_tree = [{'decisiontreeregressor__max_depth': depths,
                    'decisiontreeregressor__min_samples_leaf': num_leafs}]
```


```python
gs_tree = GridSearchCV(estimator=pipe_tree,
                       param_grid=param_grid_tree, scoring=rmse_scorer, cv=10)
best_tree = gs_tree.fit(X_val_train, y_val_train)
```


```python
print(reg_report(best_tree, X_val_test, y_val_test))
print('Best attributes: ', best_tree.best_params_)
```

    R2 score : 0.73
    MAE loss : 37877.02
    RMSE loss : 53784.42
    error % : 0.23
    None
    Best attributes:  {'decisiontreeregressor__max_depth': 11, 'decisiontreeregressor__min_samples_leaf': 20}
    

### Grid search for Linear Regression


```python
pipe_lr = make_pipeline(LinearRegression())
fit_intercept = [True, False]
normalize = [True, False]
copy_x = [True, False]

parameters = [{'linearregression__fit_intercept': fit_intercept,
               'linearregression__normalize': normalize, 'linearregression__copy_X': copy_x}]

gs_lr = GridSearchCV(estimator=pipe_lr, param_grid=parameters,
                     scoring=rmse_scorer, cv=10)
best_lr=gs_lr.fit(X_val_train, y_val_train)

```


```python
print(reg_report(best_lr, X_val_test, y_val_test))
print('Best attributes: ', best_lr.best_params_)
```

    R2 score : 0.64
    MAE loss : 45122.19
    RMSE loss : 62226.74
    error % : 0.27
    None
    Best attributes:  {'linearregression__copy_X': True, 'linearregression__fit_intercept': True, 'linearregression__normalize': True}
    

### Grid Search for Gradient Boosting Regressor


```python
ens_model = Pipeline([
    ('reg', GradientBoostingRegressor(random_state=123))
])

ens_search = GridSearchCV(
    ens_model, param_grid={
        'reg__max_depth': [number for number in np.arange(20) if number % 2 != 0],
    }
)


ens_search.fit(X_val_train, y_val_train)
ens_model = ens_search.best_estimator_
```


```python
print(reg_report(ens_search.best_estimator_, X_val_test, y_val_test))
print(ens_search.best_params_)
```

    R2 score : 0.78
    MAE loss : 33832.28
    RMSE loss : 48137.14
    error % : 0.21
    None
    {'reg__max_depth': 7}
    

### Optimizing for LightGBM Regressor


```python
def create_model(trial):
    max_depth = trial.suggest_int("max_depth", 2, 6)
    n_estimators = trial.suggest_int("n_estimators", 1, 100)
    learning_rate = trial.suggest_uniform('learning_rate', 0.0000001, 1)
    num_leaves = trial.suggest_int("num_leaves", 2, 5000)
    min_child_samples = trial.suggest_int('min_child_samples', 3, 200)
    model = LGBMRegressor(
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        max_depth=max_depth,
        num_leaves=num_leaves,
        min_child_samples=min_child_samples,
        random_state=123
    )
    return model


sampler = TPESampler(seed=123)


def objective(trial):
    model = create_model(trial)
    model.fit(X_val_train, y_val_train)
    preds = model.predict(X_val_test)
    return np.sqrt(mean_squared_error(y_val_test, preds))


study = optuna.create_study(direction="minimize", sampler=sampler)
study.optimize(objective, n_trials=400)


```

    [32m[I 2020-12-03 11:18:03,909][0m A new study created in memory with name: no-name-d284e5e2-8896-4842-899b-ca9db0f977e4[0m
    [32m[I 2020-12-03 11:18:03,949][0m Trial 0 finished with value: 49683.005937709844 and parameters: {'max_depth': 4, 'n_estimators': 93, 'learning_rate': 0.6908848859383762, 'num_leaves': 1595, 'min_child_samples': 99}. Best is trial 0 with value: 49683.005937709844.[0m
    [32m[I 2020-12-03 11:18:03,972][0m Trial 1 finished with value: 51297.007107731624 and parameters: {'max_depth': 3, 'n_estimators': 48, 'learning_rate': 0.39211757898239874, 'num_leaves': 944, 'min_child_samples': 114}. Best is trial 0 with value: 49683.005937709844.[0m
    [32m[I 2020-12-03 11:18:04,003][0m Trial 2 finished with value: 50129.70399480369 and parameters: {'max_depth': 3, 'n_estimators': 84, 'learning_rate': 0.2447593524779922, 'num_leaves': 1094, 'min_child_samples': 52}. Best is trial 0 with value: 49683.005937709844.[0m
    [32m[I 2020-12-03 11:18:04,018][0m Trial 3 finished with value: 87752.9474741871 and parameters: {'max_depth': 5, 'n_estimators': 3, 'learning_rate': 0.08372657619093589, 'num_leaves': 1348, 'min_child_samples': 87}. Best is trial 0 with value: 49683.005937709844.[0m
    [32m[I 2020-12-03 11:18:04,039][0m Trial 4 finished with value: 51195.89533543252 and parameters: {'max_depth': 2, 'n_estimators': 62, 'learning_rate': 0.8494318091346101, 'num_leaves': 1899, 'min_child_samples': 102}. Best is trial 0 with value: 49683.005937709844.[0m
    [32m[I 2020-12-03 11:18:04,096][0m Trial 5 finished with value: 49255.2081242934 and parameters: {'max_depth': 6, 'n_estimators': 93, 'learning_rate': 0.7224434103258833, 'num_leaves': 3939, 'min_child_samples': 121}. Best is trial 5 with value: 49255.2081242934.[0m
    [32m[I 2020-12-03 11:18:04,123][0m Trial 6 finished with value: 50860.18981851818 and parameters: {'max_depth': 5, 'n_estimators': 35, 'learning_rate': 0.2937141170174247, 'num_leaves': 1706, 'min_child_samples': 134}. Best is trial 5 with value: 49255.2081242934.[0m
    [32m[I 2020-12-03 11:18:04,159][0m Trial 7 finished with value: 49469.218250476995 and parameters: {'max_depth': 4, 'n_estimators': 70, 'learning_rate': 0.433701229309411, 'num_leaves': 1444, 'min_child_samples': 61}. Best is trial 5 with value: 49255.2081242934.[0m
    [32m[I 2020-12-03 11:18:04,178][0m Trial 8 finished with value: 51147.286301981156 and parameters: {'max_depth': 4, 'n_estimators': 23, 'learning_rate': 0.425830347712799, 'num_leaves': 4509, 'min_child_samples': 161}. Best is trial 5 with value: 49255.2081242934.[0m
    [32m[I 2020-12-03 11:18:04,222][0m Trial 9 finished with value: 49783.77524135198 and parameters: {'max_depth': 6, 'n_estimators': 71, 'learning_rate': 0.4642681093098624, 'num_leaves': 3228, 'min_child_samples': 137}. Best is trial 5 with value: 49255.2081242934.[0m
    [32m[I 2020-12-03 11:18:04,291][0m Trial 10 finished with value: 49855.21773842225 and parameters: {'max_depth': 6, 'n_estimators': 100, 'learning_rate': 0.9850422633953958, 'num_leaves': 4682, 'min_child_samples': 198}. Best is trial 5 with value: 49255.2081242934.[0m
    [32m[I 2020-12-03 11:18:04,324][0m Trial 11 finished with value: 55501.19612899165 and parameters: {'max_depth': 5, 'n_estimators': 77, 'learning_rate': 0.6568798190998243, 'num_leaves': 2, 'min_child_samples': 10}. Best is trial 5 with value: 49255.2081242934.[0m
    [32m[I 2020-12-03 11:18:04,363][0m Trial 12 finished with value: 50785.418238419785 and parameters: {'max_depth': 2, 'n_estimators': 99, 'learning_rate': 0.6142415946215651, 'num_leaves': 3284, 'min_child_samples': 54}. Best is trial 5 with value: 49255.2081242934.[0m
    [32m[I 2020-12-03 11:18:04,420][0m Trial 13 finished with value: 51163.672143661264 and parameters: {'max_depth': 6, 'n_estimators': 60, 'learning_rate': 0.8121454221144366, 'num_leaves': 2925, 'min_child_samples': 55}. Best is trial 5 with value: 49255.2081242934.[0m
    [32m[I 2020-12-03 11:18:04,465][0m Trial 14 finished with value: 50593.17330804066 and parameters: {'max_depth': 3, 'n_estimators': 88, 'learning_rate': 0.5638115770796578, 'num_leaves': 3876, 'min_child_samples': 13}. Best is trial 5 with value: 49255.2081242934.[0m
    [32m[I 2020-12-03 11:18:04,519][0m Trial 15 finished with value: 49145.85386346135 and parameters: {'max_depth': 5, 'n_estimators': 71, 'learning_rate': 0.7669389221980547, 'num_leaves': 94, 'min_child_samples': 66}. Best is trial 15 with value: 49145.85386346135.[0m
    [32m[I 2020-12-03 11:18:04,564][0m Trial 16 finished with value: 50289.83950172741 and parameters: {'max_depth': 5, 'n_estimators': 49, 'learning_rate': 0.9321687977372475, 'num_leaves': 83, 'min_child_samples': 174}. Best is trial 15 with value: 49145.85386346135.[0m
    [32m[I 2020-12-03 11:18:04,633][0m Trial 17 finished with value: 50925.163399725905 and parameters: {'max_depth': 6, 'n_estimators': 79, 'learning_rate': 0.7496339288687306, 'num_leaves': 3954, 'min_child_samples': 80}. Best is trial 15 with value: 49145.85386346135.[0m
    [32m[I 2020-12-03 11:18:04,683][0m Trial 18 finished with value: 50736.09731768778 and parameters: {'max_depth': 5, 'n_estimators': 61, 'learning_rate': 0.8771624203228017, 'num_leaves': 2399, 'min_child_samples': 132}. Best is trial 15 with value: 49145.85386346135.[0m
    [32m[I 2020-12-03 11:18:04,759][0m Trial 19 finished with value: 50348.55962747665 and parameters: {'max_depth': 6, 'n_estimators': 93, 'learning_rate': 0.7659563168692735, 'num_leaves': 4975, 'min_child_samples': 34}. Best is trial 15 with value: 49145.85386346135.[0m
    [32m[I 2020-12-03 11:18:04,797][0m Trial 20 finished with value: 49705.58991798905 and parameters: {'max_depth': 5, 'n_estimators': 34, 'learning_rate': 0.5557099519946296, 'num_leaves': 486, 'min_child_samples': 74}. Best is trial 15 with value: 49145.85386346135.[0m
    [32m[I 2020-12-03 11:18:04,844][0m Trial 21 finished with value: 49415.25212878373 and parameters: {'max_depth': 4, 'n_estimators': 70, 'learning_rate': 0.3244275069595957, 'num_leaves': 2311, 'min_child_samples': 66}. Best is trial 15 with value: 49145.85386346135.[0m
    [32m[I 2020-12-03 11:18:04,891][0m Trial 22 finished with value: 49956.34387186037 and parameters: {'max_depth': 4, 'n_estimators': 70, 'learning_rate': 0.20434077671327844, 'num_leaves': 2576, 'min_child_samples': 38}. Best is trial 15 with value: 49145.85386346135.[0m
    [32m[I 2020-12-03 11:18:04,949][0m Trial 23 finished with value: 82076.11272263678 and parameters: {'max_depth': 5, 'n_estimators': 56, 'learning_rate': 0.006852503668583254, 'num_leaves': 3861, 'min_child_samples': 107}. Best is trial 15 with value: 49145.85386346135.[0m
    [32m[I 2020-12-03 11:18:04,997][0m Trial 24 finished with value: 49219.08707406418 and parameters: {'max_depth': 4, 'n_estimators': 79, 'learning_rate': 0.33537041175684523, 'num_leaves': 2156, 'min_child_samples': 73}. Best is trial 15 with value: 49145.85386346135.[0m
    [32m[I 2020-12-03 11:18:05,040][0m Trial 25 finished with value: 50363.27601721342 and parameters: {'max_depth': 3, 'n_estimators': 83, 'learning_rate': 0.7165138032074955, 'num_leaves': 590, 'min_child_samples': 119}. Best is trial 15 with value: 49145.85386346135.[0m
    [32m[I 2020-12-03 11:18:05,118][0m Trial 26 finished with value: 48913.30628014194 and parameters: {'max_depth': 6, 'n_estimators': 91, 'learning_rate': 0.1522312500037789, 'num_leaves': 4287, 'min_child_samples': 31}. Best is trial 26 with value: 48913.30628014194.[0m
    [32m[I 2020-12-03 11:18:05,183][0m Trial 27 finished with value: 49173.339754501205 and parameters: {'max_depth': 5, 'n_estimators': 76, 'learning_rate': 0.14871455904751513, 'num_leaves': 1965, 'min_child_samples': 23}. Best is trial 26 with value: 48913.30628014194.[0m
    [32m[I 2020-12-03 11:18:05,252][0m Trial 28 finished with value: 48746.04254378455 and parameters: {'max_depth': 5, 'n_estimators': 89, 'learning_rate': 0.14724409989417744, 'num_leaves': 438, 'min_child_samples': 29}. Best is trial 28 with value: 48746.04254378455.[0m
    [32m[I 2020-12-03 11:18:05,361][0m Trial 29 finished with value: 50607.90152326797 and parameters: {'max_depth': 6, 'n_estimators': 100, 'learning_rate': 0.042264756003352486, 'num_leaves': 497, 'min_child_samples': 35}. Best is trial 28 with value: 48746.04254378455.[0m
    [32m[I 2020-12-03 11:18:05,435][0m Trial 30 finished with value: 49272.0536323713 and parameters: {'max_depth': 5, 'n_estimators': 91, 'learning_rate': 0.12088550304227122, 'num_leaves': 226, 'min_child_samples': 18}. Best is trial 28 with value: 48746.04254378455.[0m
    [32m[I 2020-12-03 11:18:05,509][0m Trial 31 finished with value: 49121.36784957313 and parameters: {'max_depth': 5, 'n_estimators': 86, 'learning_rate': 0.15988932570332418, 'num_leaves': 916, 'min_child_samples': 3}. Best is trial 28 with value: 48746.04254378455.[0m
    [32m[I 2020-12-03 11:18:05,580][0m Trial 32 finished with value: 48794.87922855973 and parameters: {'max_depth': 5, 'n_estimators': 88, 'learning_rate': 0.20563337244169944, 'num_leaves': 912, 'min_child_samples': 5}. Best is trial 28 with value: 48746.04254378455.[0m
    [32m[I 2020-12-03 11:18:05,674][0m Trial 33 finished with value: 48261.19607461949 and parameters: {'max_depth': 6, 'n_estimators': 87, 'learning_rate': 0.1938503447859861, 'num_leaves': 901, 'min_child_samples': 3}. Best is trial 33 with value: 48261.19607461949.[0m
    [32m[I 2020-12-03 11:18:05,753][0m Trial 34 finished with value: 48746.47457814512 and parameters: {'max_depth': 6, 'n_estimators': 98, 'learning_rate': 0.2131768059489312, 'num_leaves': 1041, 'min_child_samples': 26}. Best is trial 33 with value: 48261.19607461949.[0m
    [32m[I 2020-12-03 11:18:05,851][0m Trial 35 finished with value: 48458.65176836708 and parameters: {'max_depth': 6, 'n_estimators': 98, 'learning_rate': 0.23594174743526378, 'num_leaves': 1108, 'min_child_samples': 3}. Best is trial 33 with value: 48261.19607461949.[0m
    [32m[I 2020-12-03 11:18:05,934][0m Trial 36 finished with value: 48664.99424913521 and parameters: {'max_depth': 6, 'n_estimators': 100, 'learning_rate': 0.26764729620466965, 'num_leaves': 1218, 'min_child_samples': 43}. Best is trial 33 with value: 48261.19607461949.[0m
    [32m[I 2020-12-03 11:18:06,009][0m Trial 37 finished with value: 49045.710667719286 and parameters: {'max_depth': 6, 'n_estimators': 95, 'learning_rate': 0.2774319955115369, 'num_leaves': 1304, 'min_child_samples': 45}. Best is trial 33 with value: 48261.19607461949.[0m
    [32m[I 2020-12-03 11:18:06,071][0m Trial 38 finished with value: 49037.09812430861 and parameters: {'max_depth': 6, 'n_estimators': 82, 'learning_rate': 0.36562643132709827, 'num_leaves': 719, 'min_child_samples': 93}. Best is trial 33 with value: 48261.19607461949.[0m
    [32m[I 2020-12-03 11:18:06,171][0m Trial 39 finished with value: 49670.859422353366 and parameters: {'max_depth': 6, 'n_estimators': 100, 'learning_rate': 0.06736227792522334, 'num_leaves': 1742, 'min_child_samples': 42}. Best is trial 33 with value: 48261.19607461949.[0m
    [32m[I 2020-12-03 11:18:06,206][0m Trial 40 finished with value: 53130.34457453444 and parameters: {'max_depth': 6, 'n_estimators': 7, 'learning_rate': 0.28057781363663714, 'num_leaves': 1343, 'min_child_samples': 4}. Best is trial 33 with value: 48261.19607461949.[0m
    [32m[I 2020-12-03 11:18:06,291][0m Trial 41 finished with value: 48174.71058378774 and parameters: {'max_depth': 6, 'n_estimators': 97, 'learning_rate': 0.21218443426624783, 'num_leaves': 1171, 'min_child_samples': 22}. Best is trial 41 with value: 48174.71058378774.[0m
    [32m[I 2020-12-03 11:18:06,389][0m Trial 42 finished with value: 48793.82335134128 and parameters: {'max_depth': 6, 'n_estimators': 95, 'learning_rate': 0.10400186856160037, 'num_leaves': 1187, 'min_child_samples': 15}. Best is trial 41 with value: 48174.71058378774.[0m
    [32m[I 2020-12-03 11:18:06,464][0m Trial 43 finished with value: 48743.10234644895 and parameters: {'max_depth': 6, 'n_estimators': 96, 'learning_rate': 0.24256418250251494, 'num_leaves': 1528, 'min_child_samples': 25}. Best is trial 41 with value: 48174.71058378774.[0m
    [32m[I 2020-12-03 11:18:06,543][0m Trial 44 finished with value: 48578.63746928205 and parameters: {'max_depth': 6, 'n_estimators': 95, 'learning_rate': 0.24475510954186092, 'num_leaves': 1556, 'min_child_samples': 20}. Best is trial 41 with value: 48174.71058378774.[0m
    [32m[I 2020-12-03 11:18:06,636][0m Trial 45 finished with value: 48812.72838270048 and parameters: {'max_depth': 6, 'n_estimators': 100, 'learning_rate': 0.3964198773936517, 'num_leaves': 1650, 'min_child_samples': 11}. Best is trial 41 with value: 48174.71058378774.[0m
    [32m[I 2020-12-03 11:18:06,708][0m Trial 46 finished with value: 48488.84067411163 and parameters: {'max_depth': 6, 'n_estimators': 86, 'learning_rate': 0.4862600467680601, 'num_leaves': 862, 'min_child_samples': 47}. Best is trial 41 with value: 48174.71058378774.[0m
    [32m[I 2020-12-03 11:18:06,798][0m Trial 47 finished with value: 50288.33576222794 and parameters: {'max_depth': 6, 'n_estimators': 85, 'learning_rate': 0.4951212616747696, 'num_leaves': 796, 'min_child_samples': 3}. Best is trial 41 with value: 48174.71058378774.[0m
    [32m[I 2020-12-03 11:18:06,893][0m Trial 48 finished with value: 58603.12082021938 and parameters: {'max_depth': 6, 'n_estimators': 74, 'learning_rate': 0.01777547942573332, 'num_leaves': 1863, 'min_child_samples': 19}. Best is trial 41 with value: 48174.71058378774.[0m
    [32m[I 2020-12-03 11:18:06,943][0m Trial 49 finished with value: 49135.61732058188 and parameters: {'max_depth': 6, 'n_estimators': 42, 'learning_rate': 0.35698095140714303, 'num_leaves': 262, 'min_child_samples': 51}. Best is trial 41 with value: 48174.71058378774.[0m
    [32m[I 2020-12-03 11:18:07,027][0m Trial 50 finished with value: 49422.460897696845 and parameters: {'max_depth': 6, 'n_estimators': 82, 'learning_rate': 0.4532680362393263, 'num_leaves': 1500, 'min_child_samples': 11}. Best is trial 41 with value: 48174.71058378774.[0m
    [32m[I 2020-12-03 11:18:07,104][0m Trial 51 finished with value: 48544.70920376694 and parameters: {'max_depth': 6, 'n_estimators': 94, 'learning_rate': 0.29969978559690347, 'num_leaves': 1170, 'min_child_samples': 46}. Best is trial 41 with value: 48174.71058378774.[0m
    [32m[I 2020-12-03 11:18:07,183][0m Trial 52 finished with value: 48599.00111245134 and parameters: {'max_depth': 6, 'n_estimators': 91, 'learning_rate': 0.192549274871093, 'num_leaves': 1104, 'min_child_samples': 57}. Best is trial 41 with value: 48174.71058378774.[0m
    [32m[I 2020-12-03 11:18:07,257][0m Trial 53 finished with value: 48845.424328378576 and parameters: {'max_depth': 6, 'n_estimators': 94, 'learning_rate': 0.32306629318571484, 'num_leaves': 757, 'min_child_samples': 49}. Best is trial 41 with value: 48174.71058378774.[0m
    [32m[I 2020-12-03 11:18:07,334][0m Trial 54 finished with value: 48458.195335498945 and parameters: {'max_depth': 6, 'n_estimators': 86, 'learning_rate': 0.246461105800143, 'num_leaves': 1006, 'min_child_samples': 20}. Best is trial 41 with value: 48174.71058378774.[0m
    [32m[I 2020-12-03 11:18:07,375][0m Trial 55 finished with value: 51466.726438978825 and parameters: {'max_depth': 2, 'n_estimators': 86, 'learning_rate': 0.40659633744945706, 'num_leaves': 1036, 'min_child_samples': 36}. Best is trial 41 with value: 48174.71058378774.[0m
    [32m[I 2020-12-03 11:18:07,456][0m Trial 56 finished with value: 50749.1552675561 and parameters: {'max_depth': 6, 'n_estimators': 80, 'learning_rate': 0.5562981761964865, 'num_leaves': 676, 'min_child_samples': 8}. Best is trial 41 with value: 48174.71058378774.[0m
    [32m[I 2020-12-03 11:18:07,521][0m Trial 57 finished with value: 48784.13284820834 and parameters: {'max_depth': 6, 'n_estimators': 66, 'learning_rate': 0.30739898542926863, 'num_leaves': 272, 'min_child_samples': 28}. Best is trial 41 with value: 48174.71058378774.[0m
    [32m[I 2020-12-03 11:18:07,578][0m Trial 58 finished with value: 48988.73629644944 and parameters: {'max_depth': 5, 'n_estimators': 74, 'learning_rate': 0.23046268142847975, 'num_leaves': 878, 'min_child_samples': 59}. Best is trial 41 with value: 48174.71058378774.[0m
    [32m[I 2020-12-03 11:18:07,647][0m Trial 59 finished with value: 49832.34769069308 and parameters: {'max_depth': 6, 'n_estimators': 89, 'learning_rate': 0.6156746902709413, 'num_leaves': 1370, 'min_child_samples': 85}. Best is trial 41 with value: 48174.71058378774.[0m
    [32m[I 2020-12-03 11:18:07,699][0m Trial 60 finished with value: 48739.50948061191 and parameters: {'max_depth': 5, 'n_estimators': 66, 'learning_rate': 0.3666451937648674, 'num_leaves': 1157, 'min_child_samples': 68}. Best is trial 41 with value: 48174.71058378774.[0m
    [32m[I 2020-12-03 11:18:07,784][0m Trial 61 finished with value: 48506.36174597858 and parameters: {'max_depth': 6, 'n_estimators': 97, 'learning_rate': 0.24443561349417536, 'num_leaves': 1484, 'min_child_samples': 20}. Best is trial 41 with value: 48174.71058378774.[0m
    [32m[I 2020-12-03 11:18:07,879][0m Trial 62 finished with value: 48055.069848998566 and parameters: {'max_depth': 6, 'n_estimators': 97, 'learning_rate': 0.17935039547315995, 'num_leaves': 1009, 'min_child_samples': 15}. Best is trial 62 with value: 48055.069848998566.[0m
    [32m[I 2020-12-03 11:18:07,968][0m Trial 63 finished with value: 48094.57529687184 and parameters: {'max_depth': 6, 'n_estimators': 98, 'learning_rate': 0.18703785657087565, 'num_leaves': 622, 'min_child_samples': 16}. Best is trial 62 with value: 48055.069848998566.[0m
    [32m[I 2020-12-03 11:18:08,066][0m Trial 64 finished with value: 49475.52148561773 and parameters: {'max_depth': 6, 'n_estimators': 91, 'learning_rate': 0.07316470886034325, 'num_leaves': 406, 'min_child_samples': 12}. Best is trial 62 with value: 48055.069848998566.[0m
    [32m[I 2020-12-03 11:18:08,152][0m Trial 65 finished with value: 48678.29621856227 and parameters: {'max_depth': 6, 'n_estimators': 86, 'learning_rate': 0.11743688963964095, 'num_leaves': 569, 'min_child_samples': 14}. Best is trial 62 with value: 48055.069848998566.[0m
    [32m[I 2020-12-03 11:18:08,247][0m Trial 66 finished with value: 48415.59111777205 and parameters: {'max_depth': 6, 'n_estimators': 97, 'learning_rate': 0.17893477909415656, 'num_leaves': 948, 'min_child_samples': 4}. Best is trial 62 with value: 48055.069848998566.[0m
    [32m[I 2020-12-03 11:18:08,345][0m Trial 67 finished with value: 48528.448306271595 and parameters: {'max_depth': 6, 'n_estimators': 98, 'learning_rate': 0.1791594255194775, 'num_leaves': 1056, 'min_child_samples': 3}. Best is trial 62 with value: 48055.069848998566.[0m
    [32m[I 2020-12-03 11:18:08,404][0m Trial 68 finished with value: 50408.69771027497 and parameters: {'max_depth': 5, 'n_estimators': 91, 'learning_rate': 0.12875789956560327, 'num_leaves': 649, 'min_child_samples': 158}. Best is trial 62 with value: 48055.069848998566.[0m
    [32m[I 2020-12-03 11:18:08,496][0m Trial 69 finished with value: 47828.34314612498 and parameters: {'max_depth': 6, 'n_estimators': 100, 'learning_rate': 0.18437718773969192, 'num_leaves': 336, 'min_child_samples': 8}. Best is trial 69 with value: 47828.34314612498.[0m
    [32m[I 2020-12-03 11:18:08,584][0m Trial 70 finished with value: 49259.69477131353 and parameters: {'max_depth': 6, 'n_estimators': 92, 'learning_rate': 0.09498205461234376, 'num_leaves': 350, 'min_child_samples': 32}. Best is trial 69 with value: 47828.34314612498.[0m
    [32m[I 2020-12-03 11:18:08,674][0m Trial 71 finished with value: 48704.79953036234 and parameters: {'max_depth': 6, 'n_estimators': 98, 'learning_rate': 0.21199097961022692, 'num_leaves': 942, 'min_child_samples': 8}. Best is trial 69 with value: 47828.34314612498.[0m
    [32m[I 2020-12-03 11:18:08,774][0m Trial 72 finished with value: 48027.46687094709 and parameters: {'max_depth': 6, 'n_estimators': 100, 'learning_rate': 0.1748177080860288, 'num_leaves': 638, 'min_child_samples': 3}. Best is trial 69 with value: 47828.34314612498.[0m
    [32m[I 2020-12-03 11:18:08,856][0m Trial 73 finished with value: 48262.207577645466 and parameters: {'max_depth': 6, 'n_estimators': 89, 'learning_rate': 0.17601727560933747, 'num_leaves': 558, 'min_child_samples': 23}. Best is trial 69 with value: 47828.34314612498.[0m
    [32m[I 2020-12-03 11:18:08,942][0m Trial 74 finished with value: 48316.46677342675 and parameters: {'max_depth': 6, 'n_estimators': 100, 'learning_rate': 0.17168424336541707, 'num_leaves': 165, 'min_child_samples': 15}. Best is trial 69 with value: 47828.34314612498.[0m
    [32m[I 2020-12-03 11:18:08,998][0m Trial 75 finished with value: 51897.34236940322 and parameters: {'max_depth': 6, 'n_estimators': 89, 'learning_rate': 0.05449348424514333, 'num_leaves': 11, 'min_child_samples': 25}. Best is trial 69 with value: 47828.34314612498.[0m
    [32m[I 2020-12-03 11:18:09,044][0m Trial 76 finished with value: 50931.80071813376 and parameters: {'max_depth': 6, 'n_estimators': 93, 'learning_rate': 0.1546588023930376, 'num_leaves': 6, 'min_child_samples': 17}. Best is trial 69 with value: 47828.34314612498.[0m
    [32m[I 2020-12-03 11:18:09,160][0m Trial 77 finished with value: 50725.85535828569 and parameters: {'max_depth': 6, 'n_estimators': 100, 'learning_rate': 0.035294573566607396, 'num_leaves': 152, 'min_child_samples': 9}. Best is trial 69 with value: 47828.34314612498.[0m
    [32m[I 2020-12-03 11:18:09,229][0m Trial 78 finished with value: 48899.79335980182 and parameters: {'max_depth': 5, 'n_estimators': 100, 'learning_rate': 0.17551148388012797, 'num_leaves': 502, 'min_child_samples': 30}. Best is trial 69 with value: 47828.34314612498.[0m
    [32m[I 2020-12-03 11:18:09,272][0m Trial 79 finished with value: 54656.09819108059 and parameters: {'max_depth': 6, 'n_estimators': 18, 'learning_rate': 0.09798863775780085, 'num_leaves': 141, 'min_child_samples': 37}. Best is trial 69 with value: 47828.34314612498.[0m
    [32m[I 2020-12-03 11:18:09,340][0m Trial 80 finished with value: 49275.97756747049 and parameters: {'max_depth': 6, 'n_estimators': 56, 'learning_rate': 0.1268959428244691, 'num_leaves': 378, 'min_child_samples': 23}. Best is trial 69 with value: 47828.34314612498.[0m
    [32m[I 2020-12-03 11:18:09,434][0m Trial 81 finished with value: 48352.07517482678 and parameters: {'max_depth': 6, 'n_estimators': 96, 'learning_rate': 0.17730117750092794, 'num_leaves': 633, 'min_child_samples': 15}. Best is trial 69 with value: 47828.34314612498.[0m
    [32m[I 2020-12-03 11:18:09,522][0m Trial 82 finished with value: 48388.41310255237 and parameters: {'max_depth': 6, 'n_estimators': 95, 'learning_rate': 0.19831734236083606, 'num_leaves': 630, 'min_child_samples': 16}. Best is trial 69 with value: 47828.34314612498.[0m
    [32m[I 2020-12-03 11:18:09,625][0m Trial 83 finished with value: 47954.045949281564 and parameters: {'max_depth': 6, 'n_estimators': 100, 'learning_rate': 0.13833812356111508, 'num_leaves': 547, 'min_child_samples': 7}. Best is trial 69 with value: 47828.34314612498.[0m
    [32m[I 2020-12-03 11:18:09,723][0m Trial 84 finished with value: 48297.365078151764 and parameters: {'max_depth': 6, 'n_estimators': 100, 'learning_rate': 0.14307181112611939, 'num_leaves': 471, 'min_child_samples': 8}. Best is trial 69 with value: 47828.34314612498.[0m
    [32m[I 2020-12-03 11:18:09,811][0m Trial 85 finished with value: 48323.97234362703 and parameters: {'max_depth': 6, 'n_estimators': 89, 'learning_rate': 0.13804097068744797, 'num_leaves': 814, 'min_child_samples': 7}. Best is trial 69 with value: 47828.34314612498.[0m
    [32m[I 2020-12-03 11:18:09,857][0m Trial 86 finished with value: 52311.71324732158 and parameters: {'max_depth': 3, 'n_estimators': 93, 'learning_rate': 0.0770086061305673, 'num_leaves': 517, 'min_child_samples': 9}. Best is trial 69 with value: 47828.34314612498.[0m
    [32m[I 2020-12-03 11:18:09,940][0m Trial 87 finished with value: 48621.15800197164 and parameters: {'max_depth': 6, 'n_estimators': 97, 'learning_rate': 0.1446355924828832, 'num_leaves': 747, 'min_child_samples': 24}. Best is trial 69 with value: 47828.34314612498.[0m
    [32m[I 2020-12-03 11:18:10,038][0m Trial 88 finished with value: 47456.40122651236 and parameters: {'max_depth': 6, 'n_estimators': 100, 'learning_rate': 0.22222685669158865, 'num_leaves': 325, 'min_child_samples': 4}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:10,104][0m Trial 89 finished with value: 48584.74834406959 and parameters: {'max_depth': 5, 'n_estimators': 83, 'learning_rate': 0.25605794729138426, 'num_leaves': 336, 'min_child_samples': 3}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:10,181][0m Trial 90 finished with value: 48890.0965593056 and parameters: {'max_depth': 6, 'n_estimators': 88, 'learning_rate': 0.21927613316264347, 'num_leaves': 1264, 'min_child_samples': 21}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:10,271][0m Trial 91 finished with value: 48657.658786568325 and parameters: {'max_depth': 6, 'n_estimators': 100, 'learning_rate': 0.1978918317117621, 'num_leaves': 464, 'min_child_samples': 12}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:10,359][0m Trial 92 finished with value: 47855.92841618049 and parameters: {'max_depth': 6, 'n_estimators': 98, 'learning_rate': 0.2728717346186586, 'num_leaves': 282, 'min_child_samples': 7}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:10,423][0m Trial 93 finished with value: 50100.27958216616 and parameters: {'max_depth': 6, 'n_estimators': 93, 'learning_rate': 0.22313960892691567, 'num_leaves': 321, 'min_child_samples': 196}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:10,486][0m Trial 94 finished with value: 48693.028742715906 and parameters: {'max_depth': 6, 'n_estimators': 96, 'learning_rate': 0.2645399637290214, 'num_leaves': 16, 'min_child_samples': 41}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:10,583][0m Trial 95 finished with value: 48061.64038097732 and parameters: {'max_depth': 6, 'n_estimators': 97, 'learning_rate': 0.27086478394051505, 'num_leaves': 224, 'min_child_samples': 3}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:10,684][0m Trial 96 finished with value: 48496.57921072749 and parameters: {'max_depth': 6, 'n_estimators': 97, 'learning_rate': 0.27622459131695354, 'num_leaves': 222, 'min_child_samples': 3}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:10,780][0m Trial 97 finished with value: 48848.997853823625 and parameters: {'max_depth': 6, 'n_estimators': 94, 'learning_rate': 0.2930171852605864, 'num_leaves': 732, 'min_child_samples': 6}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:10,885][0m Trial 98 finished with value: 48920.633024350485 and parameters: {'max_depth': 6, 'n_estimators': 98, 'learning_rate': 0.3308464794108949, 'num_leaves': 80, 'min_child_samples': 12}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:10,989][0m Trial 99 finished with value: 48245.80356258332 and parameters: {'max_depth': 6, 'n_estimators': 100, 'learning_rate': 0.22815699606529374, 'num_leaves': 827, 'min_child_samples': 3}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:11,078][0m Trial 100 finished with value: 49031.36439280463 and parameters: {'max_depth': 6, 'n_estimators': 99, 'learning_rate': 0.1138393705421964, 'num_leaves': 2927, 'min_child_samples': 28}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:11,169][0m Trial 101 finished with value: 48237.24577776946 and parameters: {'max_depth': 6, 'n_estimators': 95, 'learning_rate': 0.23213481059635505, 'num_leaves': 278, 'min_child_samples': 6}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:11,272][0m Trial 102 finished with value: 48415.41051726137 and parameters: {'max_depth': 6, 'n_estimators': 95, 'learning_rate': 0.2277612584408959, 'num_leaves': 247, 'min_child_samples': 17}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:11,365][0m Trial 103 finished with value: 48614.26261417661 and parameters: {'max_depth': 6, 'n_estimators': 91, 'learning_rate': 0.2990946842045832, 'num_leaves': 390, 'min_child_samples': 7}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:11,474][0m Trial 104 finished with value: 48522.21720528898 and parameters: {'max_depth': 6, 'n_estimators': 100, 'learning_rate': 0.2615596905925878, 'num_leaves': 596, 'min_child_samples': 3}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:11,559][0m Trial 105 finished with value: 48128.88830563071 and parameters: {'max_depth': 6, 'n_estimators': 98, 'learning_rate': 0.2099390159870803, 'num_leaves': 805, 'min_child_samples': 13}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:11,648][0m Trial 106 finished with value: 48392.40701744061 and parameters: {'max_depth': 6, 'n_estimators': 92, 'learning_rate': 0.19685523975291944, 'num_leaves': 98, 'min_child_samples': 11}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:11,733][0m Trial 107 finished with value: 48098.2161392496 and parameters: {'max_depth': 6, 'n_estimators': 96, 'learning_rate': 0.2054496547555138, 'num_leaves': 250, 'min_child_samples': 16}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:11,819][0m Trial 108 finished with value: 47984.638223141905 and parameters: {'max_depth': 6, 'n_estimators': 98, 'learning_rate': 0.15229151263999252, 'num_leaves': 707, 'min_child_samples': 19}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:11,902][0m Trial 109 finished with value: 48226.752473668756 and parameters: {'max_depth': 6, 'n_estimators': 98, 'learning_rate': 0.16261418405770828, 'num_leaves': 419, 'min_child_samples': 18}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:11,992][0m Trial 110 finished with value: 49154.503249395406 and parameters: {'max_depth': 6, 'n_estimators': 93, 'learning_rate': 0.08980309168040479, 'num_leaves': 728, 'min_child_samples': 31}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:12,083][0m Trial 111 finished with value: 48680.17172722768 and parameters: {'max_depth': 6, 'n_estimators': 97, 'learning_rate': 0.15845973630991728, 'num_leaves': 585, 'min_child_samples': 14}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:12,164][0m Trial 112 finished with value: 48158.01391932452 and parameters: {'max_depth': 6, 'n_estimators': 96, 'learning_rate': 0.21101767800637855, 'num_leaves': 960, 'min_child_samples': 20}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:12,246][0m Trial 113 finished with value: 48458.66748782437 and parameters: {'max_depth': 6, 'n_estimators': 95, 'learning_rate': 0.20711961171842702, 'num_leaves': 965, 'min_child_samples': 19}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:12,323][0m Trial 114 finished with value: 48896.782631588096 and parameters: {'max_depth': 6, 'n_estimators': 90, 'learning_rate': 0.18835168425969065, 'num_leaves': 182, 'min_child_samples': 25}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:12,412][0m Trial 115 finished with value: 48331.40250289436 and parameters: {'max_depth': 6, 'n_estimators': 98, 'learning_rate': 0.2562666744327625, 'num_leaves': 688, 'min_child_samples': 13}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:12,501][0m Trial 116 finished with value: 48901.07878293084 and parameters: {'max_depth': 6, 'n_estimators': 100, 'learning_rate': 0.10903952751844782, 'num_leaves': 521, 'min_child_samples': 27}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:12,586][0m Trial 117 finished with value: 48534.764436526544 and parameters: {'max_depth': 6, 'n_estimators': 94, 'learning_rate': 0.351533637922205, 'num_leaves': 815, 'min_child_samples': 10}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:12,667][0m Trial 118 finished with value: 48056.15661917248 and parameters: {'max_depth': 6, 'n_estimators': 92, 'learning_rate': 0.28318563007257813, 'num_leaves': 352, 'min_child_samples': 17}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:12,718][0m Trial 119 finished with value: 49369.88997236925 and parameters: {'max_depth': 6, 'n_estimators': 35, 'learning_rate': 0.31692470760407343, 'num_leaves': 356, 'min_child_samples': 16}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:12,770][0m Trial 120 finished with value: 49311.43309110832 and parameters: {'max_depth': 6, 'n_estimators': 40, 'learning_rate': 0.2902453041563355, 'num_leaves': 432, 'min_child_samples': 33}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:12,855][0m Trial 121 finished with value: 48647.16898095133 and parameters: {'max_depth': 6, 'n_estimators': 98, 'learning_rate': 0.16085953758800506, 'num_leaves': 48, 'min_child_samples': 21}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:12,950][0m Trial 122 finished with value: 48424.79015399273 and parameters: {'max_depth': 6, 'n_estimators': 96, 'learning_rate': 0.12677830639162593, 'num_leaves': 267, 'min_child_samples': 10}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:13,032][0m Trial 123 finished with value: 48313.30319760357 and parameters: {'max_depth': 6, 'n_estimators': 92, 'learning_rate': 0.24779023438376976, 'num_leaves': 670, 'min_child_samples': 18}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:13,123][0m Trial 124 finished with value: 48243.130437966276 and parameters: {'max_depth': 6, 'n_estimators': 100, 'learning_rate': 0.27706832974470197, 'num_leaves': 529, 'min_child_samples': 7}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:13,211][0m Trial 125 finished with value: 48536.6975247313 and parameters: {'max_depth': 6, 'n_estimators': 96, 'learning_rate': 0.20730943438958016, 'num_leaves': 847, 'min_child_samples': 14}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:13,292][0m Trial 126 finished with value: 48570.3580972444 and parameters: {'max_depth': 6, 'n_estimators': 93, 'learning_rate': 0.17855333261968298, 'num_leaves': 174, 'min_child_samples': 23}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:13,397][0m Trial 127 finished with value: 48239.17465170596 and parameters: {'max_depth': 6, 'n_estimators': 100, 'learning_rate': 0.13514817349530872, 'num_leaves': 315, 'min_child_samples': 3}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:13,489][0m Trial 128 finished with value: 48240.51924988144 and parameters: {'max_depth': 6, 'n_estimators': 98, 'learning_rate': 0.23666381862741118, 'num_leaves': 1069, 'min_child_samples': 12}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:13,561][0m Trial 129 finished with value: 49732.38706793548 and parameters: {'max_depth': 6, 'n_estimators': 90, 'learning_rate': 0.18850514060853668, 'num_leaves': 461, 'min_child_samples': 107}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:13,644][0m Trial 130 finished with value: 48700.60577024878 and parameters: {'max_depth': 6, 'n_estimators': 95, 'learning_rate': 0.15230445927316733, 'num_leaves': 604, 'min_child_samples': 39}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:13,727][0m Trial 131 finished with value: 48244.68429755912 and parameters: {'max_depth': 6, 'n_estimators': 97, 'learning_rate': 0.20162365462270218, 'num_leaves': 941, 'min_child_samples': 19}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:13,808][0m Trial 132 finished with value: 48423.699792375715 and parameters: {'max_depth': 6, 'n_estimators': 100, 'learning_rate': 0.21584171179679534, 'num_leaves': 1224, 'min_child_samples': 27}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:13,891][0m Trial 133 finished with value: 48441.40072172274 and parameters: {'max_depth': 6, 'n_estimators': 98, 'learning_rate': 0.26772742004449734, 'num_leaves': 1386, 'min_child_samples': 21}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:13,984][0m Trial 134 finished with value: 48130.897732882986 and parameters: {'max_depth': 6, 'n_estimators': 94, 'learning_rate': 0.1679438152335715, 'num_leaves': 791, 'min_child_samples': 6}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:14,077][0m Trial 135 finished with value: 48073.70524937009 and parameters: {'max_depth': 6, 'n_estimators': 93, 'learning_rate': 0.1590090400581533, 'num_leaves': 744, 'min_child_samples': 7}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:14,182][0m Trial 136 finished with value: 49908.574104749045 and parameters: {'max_depth': 6, 'n_estimators': 88, 'learning_rate': 0.05733929838406211, 'num_leaves': 752, 'min_child_samples': 7}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:14,277][0m Trial 137 finished with value: 48325.0941374982 and parameters: {'max_depth': 6, 'n_estimators': 91, 'learning_rate': 0.17167938921466633, 'num_leaves': 625, 'min_child_samples': 3}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:14,371][0m Trial 138 finished with value: 48126.18909507445 and parameters: {'max_depth': 6, 'n_estimators': 93, 'learning_rate': 0.13788351281741096, 'num_leaves': 381, 'min_child_samples': 8}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:14,465][0m Trial 139 finished with value: 48640.14051813394 and parameters: {'max_depth': 6, 'n_estimators': 93, 'learning_rate': 0.11878873929317439, 'num_leaves': 385, 'min_child_samples': 15}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:14,567][0m Trial 140 finished with value: 48875.506934744786 and parameters: {'max_depth': 6, 'n_estimators': 100, 'learning_rate': 0.08901944604835775, 'num_leaves': 229, 'min_child_samples': 11}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:14,662][0m Trial 141 finished with value: 48159.241958800594 and parameters: {'max_depth': 6, 'n_estimators': 94, 'learning_rate': 0.14603666591470152, 'num_leaves': 538, 'min_child_samples': 6}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:14,760][0m Trial 142 finished with value: 48356.6286964388 and parameters: {'max_depth': 6, 'n_estimators': 92, 'learning_rate': 0.16371449479089142, 'num_leaves': 463, 'min_child_samples': 3}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:14,856][0m Trial 143 finished with value: 48161.30118012038 and parameters: {'max_depth': 6, 'n_estimators': 95, 'learning_rate': 0.13168704014893134, 'num_leaves': 705, 'min_child_samples': 9}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:14,943][0m Trial 144 finished with value: 48052.32704350273 and parameters: {'max_depth': 6, 'n_estimators': 90, 'learning_rate': 0.18735278233646352, 'num_leaves': 110, 'min_child_samples': 14}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:15,022][0m Trial 145 finished with value: 48311.23167819065 and parameters: {'max_depth': 6, 'n_estimators': 84, 'learning_rate': 0.18747299146150478, 'num_leaves': 102, 'min_child_samples': 16}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:15,096][0m Trial 146 finished with value: 48421.35088572362 and parameters: {'max_depth': 6, 'n_estimators': 91, 'learning_rate': 0.23272092627716545, 'num_leaves': 26, 'min_child_samples': 13}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:15,194][0m Trial 147 finished with value: 48645.528022122155 and parameters: {'max_depth': 6, 'n_estimators': 98, 'learning_rate': 0.09855422794588214, 'num_leaves': 275, 'min_child_samples': 9}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:15,251][0m Trial 148 finished with value: 49249.63575626118 and parameters: {'max_depth': 4, 'n_estimators': 87, 'learning_rate': 0.24878939595103877, 'num_leaves': 377, 'min_child_samples': 3}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:15,349][0m Trial 149 finished with value: 48452.120586146964 and parameters: {'max_depth': 6, 'n_estimators': 89, 'learning_rate': 0.18927636195016614, 'num_leaves': 137, 'min_child_samples': 16}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:15,450][0m Trial 150 finished with value: 48497.40654271524 and parameters: {'max_depth': 6, 'n_estimators': 100, 'learning_rate': 0.13413884064149104, 'num_leaves': 218, 'min_child_samples': 11}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:15,544][0m Trial 151 finished with value: 48128.512021014496 and parameters: {'max_depth': 6, 'n_estimators': 94, 'learning_rate': 0.16377819862184834, 'num_leaves': 855, 'min_child_samples': 6}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:15,634][0m Trial 152 finished with value: 48490.16231126513 and parameters: {'max_depth': 6, 'n_estimators': 96, 'learning_rate': 0.2167987773163239, 'num_leaves': 555, 'min_child_samples': 8}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:15,725][0m Trial 153 finished with value: 48358.81257461371 and parameters: {'max_depth': 6, 'n_estimators': 98, 'learning_rate': 0.1517984703243249, 'num_leaves': 326, 'min_child_samples': 15}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:15,825][0m Trial 154 finished with value: 48249.448262949634 and parameters: {'max_depth': 6, 'n_estimators': 93, 'learning_rate': 0.18164436365914266, 'num_leaves': 446, 'min_child_samples': 3}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:15,867][0m Trial 155 finished with value: 51632.1427046834 and parameters: {'max_depth': 6, 'n_estimators': 96, 'learning_rate': 0.5332531635660139, 'num_leaves': 3, 'min_child_samples': 131}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:15,954][0m Trial 156 finished with value: 48951.55955618927 and parameters: {'max_depth': 6, 'n_estimators': 90, 'learning_rate': 0.11176872025068602, 'num_leaves': 3506, 'min_child_samples': 23}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:16,047][0m Trial 157 finished with value: 48567.01223996157 and parameters: {'max_depth': 6, 'n_estimators': 100, 'learning_rate': 0.15526758124634632, 'num_leaves': 877, 'min_child_samples': 11}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:16,145][0m Trial 158 finished with value: 48037.5025399811 and parameters: {'max_depth': 6, 'n_estimators': 94, 'learning_rate': 0.19968716817235915, 'num_leaves': 658, 'min_child_samples': 3}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:16,246][0m Trial 159 finished with value: 48233.48885123259 and parameters: {'max_depth': 6, 'n_estimators': 94, 'learning_rate': 0.19666596525223934, 'num_leaves': 637, 'min_child_samples': 3}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:16,332][0m Trial 160 finished with value: 48520.778802422676 and parameters: {'max_depth': 6, 'n_estimators': 86, 'learning_rate': 0.2292717664624708, 'num_leaves': 501, 'min_child_samples': 6}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:16,424][0m Trial 161 finished with value: 48705.71048678502 and parameters: {'max_depth': 6, 'n_estimators': 97, 'learning_rate': 0.17167431376151576, 'num_leaves': 739, 'min_child_samples': 13}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:16,508][0m Trial 162 finished with value: 48065.8392418092 and parameters: {'max_depth': 6, 'n_estimators': 91, 'learning_rate': 0.21310907597794732, 'num_leaves': 659, 'min_child_samples': 18}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:16,591][0m Trial 163 finished with value: 48281.36739368896 and parameters: {'max_depth': 6, 'n_estimators': 92, 'learning_rate': 0.24568888179182805, 'num_leaves': 347, 'min_child_samples': 19}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:16,683][0m Trial 164 finished with value: 48431.103467014924 and parameters: {'max_depth': 6, 'n_estimators': 89, 'learning_rate': 0.14515658697115058, 'num_leaves': 595, 'min_child_samples': 8}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:16,780][0m Trial 165 finished with value: 48578.544054715014 and parameters: {'max_depth': 6, 'n_estimators': 95, 'learning_rate': 0.2733130163603263, 'num_leaves': 664, 'min_child_samples': 3}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:16,863][0m Trial 166 finished with value: 48207.66915106244 and parameters: {'max_depth': 6, 'n_estimators': 91, 'learning_rate': 0.19334619557625954, 'num_leaves': 407, 'min_child_samples': 17}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:16,912][0m Trial 167 finished with value: 49937.92727751003 and parameters: {'max_depth': 6, 'n_estimators': 27, 'learning_rate': 0.21656321618875615, 'num_leaves': 179, 'min_child_samples': 26}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:17,006][0m Trial 168 finished with value: 48271.7719522395 and parameters: {'max_depth': 6, 'n_estimators': 93, 'learning_rate': 0.17365087323639342, 'num_leaves': 517, 'min_child_samples': 7}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:17,097][0m Trial 169 finished with value: 48724.02865302174 and parameters: {'max_depth': 6, 'n_estimators': 87, 'learning_rate': 0.12622637695131236, 'num_leaves': 306, 'min_child_samples': 11}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:17,198][0m Trial 170 finished with value: 49266.33077009546 and parameters: {'max_depth': 6, 'n_estimators': 96, 'learning_rate': 0.07375475116904978, 'num_leaves': 703, 'min_child_samples': 20}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:17,288][0m Trial 171 finished with value: 48105.98787700684 and parameters: {'max_depth': 6, 'n_estimators': 98, 'learning_rate': 0.21192575521618331, 'num_leaves': 872, 'min_child_samples': 14}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:17,338][0m Trial 172 finished with value: 49968.90212809831 and parameters: {'max_depth': 3, 'n_estimators': 99, 'learning_rate': 0.22306041437769641, 'num_leaves': 866, 'min_child_samples': 14}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:17,433][0m Trial 173 finished with value: 48282.47891368974 and parameters: {'max_depth': 6, 'n_estimators': 94, 'learning_rate': 0.16189838244823762, 'num_leaves': 792, 'min_child_samples': 6}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:17,519][0m Trial 174 finished with value: 48514.76761458828 and parameters: {'max_depth': 6, 'n_estimators': 98, 'learning_rate': 0.19197404299107204, 'num_leaves': 949, 'min_child_samples': 23}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:17,601][0m Trial 175 finished with value: 48591.649583238075 and parameters: {'max_depth': 6, 'n_estimators': 96, 'learning_rate': 0.25308556275270694, 'num_leaves': 441, 'min_child_samples': 17}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:17,704][0m Trial 176 finished with value: 49382.02884744737 and parameters: {'max_depth': 6, 'n_estimators': 100, 'learning_rate': 0.2997469567823079, 'num_leaves': 1018, 'min_child_samples': 3}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:17,789][0m Trial 177 finished with value: 48464.54594915011 and parameters: {'max_depth': 6, 'n_estimators': 90, 'learning_rate': 0.20545229397172715, 'num_leaves': 593, 'min_child_samples': 9}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:17,871][0m Trial 178 finished with value: 48887.703424946245 and parameters: {'max_depth': 6, 'n_estimators': 92, 'learning_rate': 0.14116822633978504, 'num_leaves': 121, 'min_child_samples': 30}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:17,959][0m Trial 179 finished with value: 48611.29314496615 and parameters: {'max_depth': 6, 'n_estimators': 96, 'learning_rate': 0.2301075402223084, 'num_leaves': 240, 'min_child_samples': 12}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:18,051][0m Trial 180 finished with value: 48467.71929068792 and parameters: {'max_depth': 6, 'n_estimators': 100, 'learning_rate': 0.17889061277355897, 'num_leaves': 663, 'min_child_samples': 18}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:18,142][0m Trial 181 finished with value: 48351.70641982152 and parameters: {'max_depth': 6, 'n_estimators': 98, 'learning_rate': 0.20287118593055126, 'num_leaves': 799, 'min_child_samples': 13}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:18,235][0m Trial 182 finished with value: 48704.73639917341 and parameters: {'max_depth': 6, 'n_estimators': 94, 'learning_rate': 0.21225276859560607, 'num_leaves': 1106, 'min_child_samples': 9}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:18,268][0m Trial 183 finished with value: 75171.64718013453 and parameters: {'max_depth': 6, 'n_estimators': 3, 'learning_rate': 0.16189652848594982, 'num_leaves': 849, 'min_child_samples': 3}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:18,355][0m Trial 184 finished with value: 48407.241684701854 and parameters: {'max_depth': 6, 'n_estimators': 98, 'learning_rate': 0.28177257447570503, 'num_leaves': 513, 'min_child_samples': 15}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:18,447][0m Trial 185 finished with value: 48522.49395033572 and parameters: {'max_depth': 6, 'n_estimators': 96, 'learning_rate': 0.24142403668718565, 'num_leaves': 728, 'min_child_samples': 6}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:18,533][0m Trial 186 finished with value: 48205.674903380466 and parameters: {'max_depth': 6, 'n_estimators': 100, 'learning_rate': 0.18442490198170963, 'num_leaves': 579, 'min_child_samples': 22}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:18,625][0m Trial 187 finished with value: 48382.16643707315 and parameters: {'max_depth': 6, 'n_estimators': 94, 'learning_rate': 0.14823552288780145, 'num_leaves': 936, 'min_child_samples': 9}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:18,725][0m Trial 188 finished with value: 48545.65954381285 and parameters: {'max_depth': 6, 'n_estimators': 98, 'learning_rate': 0.10883703920171384, 'num_leaves': 329, 'min_child_samples': 14}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:18,806][0m Trial 189 finished with value: 48863.757705747754 and parameters: {'max_depth': 6, 'n_estimators': 92, 'learning_rate': 0.26193012654985, 'num_leaves': 433, 'min_child_samples': 19}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:18,907][0m Trial 190 finished with value: 56821.97117392407 and parameters: {'max_depth': 6, 'n_estimators': 97, 'learning_rate': 0.9860173984653513, 'num_leaves': 2195, 'min_child_samples': 3}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:19,004][0m Trial 191 finished with value: 48511.14653282662 and parameters: {'max_depth': 6, 'n_estimators': 94, 'learning_rate': 0.16870043824021397, 'num_leaves': 793, 'min_child_samples': 6}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:19,094][0m Trial 192 finished with value: 48326.89565190385 and parameters: {'max_depth': 6, 'n_estimators': 95, 'learning_rate': 0.20008850605941508, 'num_leaves': 797, 'min_child_samples': 11}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:19,195][0m Trial 193 finished with value: 48122.35463198356 and parameters: {'max_depth': 6, 'n_estimators': 100, 'learning_rate': 0.16624833400312122, 'num_leaves': 624, 'min_child_samples': 7}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:19,298][0m Trial 194 finished with value: 48194.367288435074 and parameters: {'max_depth': 6, 'n_estimators': 100, 'learning_rate': 0.1241746658205232, 'num_leaves': 593, 'min_child_samples': 11}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:19,386][0m Trial 195 finished with value: 48666.41474513909 and parameters: {'max_depth': 6, 'n_estimators': 98, 'learning_rate': 0.22098138023102418, 'num_leaves': 679, 'min_child_samples': 16}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:19,493][0m Trial 196 finished with value: 48609.86594830931 and parameters: {'max_depth': 6, 'n_estimators': 100, 'learning_rate': 0.1865794850075113, 'num_leaves': 461, 'min_child_samples': 3}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:19,591][0m Trial 197 finished with value: 48244.873879552055 and parameters: {'max_depth': 6, 'n_estimators': 96, 'learning_rate': 0.15335362508159572, 'num_leaves': 365, 'min_child_samples': 7}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:19,679][0m Trial 198 finished with value: 48100.07770079959 and parameters: {'max_depth': 6, 'n_estimators': 91, 'learning_rate': 0.210867313672899, 'num_leaves': 231, 'min_child_samples': 13}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:19,762][0m Trial 199 finished with value: 48251.16263500401 and parameters: {'max_depth': 6, 'n_estimators': 90, 'learning_rate': 0.17437830667156257, 'num_leaves': 226, 'min_child_samples': 21}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:19,855][0m Trial 200 finished with value: 48589.082331861195 and parameters: {'max_depth': 6, 'n_estimators': 92, 'learning_rate': 0.13864248484839659, 'num_leaves': 253, 'min_child_samples': 10}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:19,939][0m Trial 201 finished with value: 48515.177987193376 and parameters: {'max_depth': 6, 'n_estimators': 88, 'learning_rate': 0.20683119668696073, 'num_leaves': 2545, 'min_child_samples': 14}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:20,035][0m Trial 202 finished with value: 47928.77216939156 and parameters: {'max_depth': 6, 'n_estimators': 98, 'learning_rate': 0.2237485969310101, 'num_leaves': 92, 'min_child_samples': 7}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:20,136][0m Trial 203 finished with value: 48430.95377226762 and parameters: {'max_depth': 6, 'n_estimators': 93, 'learning_rate': 0.22947325279171765, 'num_leaves': 140, 'min_child_samples': 3}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:20,201][0m Trial 204 finished with value: 48800.57226930227 and parameters: {'max_depth': 6, 'n_estimators': 96, 'learning_rate': 0.24862196756550692, 'num_leaves': 16, 'min_child_samples': 8}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:20,288][0m Trial 205 finished with value: 48145.053398021555 and parameters: {'max_depth': 6, 'n_estimators': 100, 'learning_rate': 0.1890518979195632, 'num_leaves': 75, 'min_child_samples': 17}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:20,383][0m Trial 206 finished with value: 48317.85517146324 and parameters: {'max_depth': 6, 'n_estimators': 91, 'learning_rate': 0.1636146867417246, 'num_leaves': 287, 'min_child_samples': 6}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:20,485][0m Trial 207 finished with value: 48308.66614823731 and parameters: {'max_depth': 6, 'n_estimators': 97, 'learning_rate': 0.21887404377627934, 'num_leaves': 402, 'min_child_samples': 3}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:20,579][0m Trial 208 finished with value: 48239.048185644366 and parameters: {'max_depth': 6, 'n_estimators': 95, 'learning_rate': 0.1917778368398247, 'num_leaves': 130, 'min_child_samples': 11}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:20,667][0m Trial 209 finished with value: 48745.194323216776 and parameters: {'max_depth': 6, 'n_estimators': 99, 'learning_rate': 0.23727580096765943, 'num_leaves': 497, 'min_child_samples': 14}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:20,720][0m Trial 210 finished with value: 50056.74688876157 and parameters: {'max_depth': 6, 'n_estimators': 55, 'learning_rate': 0.2656645319752698, 'num_leaves': 14, 'min_child_samples': 156}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:20,815][0m Trial 211 finished with value: 48112.397639515264 and parameters: {'max_depth': 6, 'n_estimators': 98, 'learning_rate': 0.20702065462102856, 'num_leaves': 671, 'min_child_samples': 10}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:20,912][0m Trial 212 finished with value: 48221.69632820157 and parameters: {'max_depth': 6, 'n_estimators': 98, 'learning_rate': 0.16849742951783456, 'num_leaves': 646, 'min_child_samples': 7}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:21,000][0m Trial 213 finished with value: 48464.39368831667 and parameters: {'max_depth': 6, 'n_estimators': 94, 'learning_rate': 0.20395016595400173, 'num_leaves': 580, 'min_child_samples': 10}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:21,092][0m Trial 214 finished with value: 48191.006787883714 and parameters: {'max_depth': 6, 'n_estimators': 100, 'learning_rate': 0.14084726711621146, 'num_leaves': 354, 'min_child_samples': 18}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:21,137][0m Trial 215 finished with value: 52305.92391444521 and parameters: {'max_depth': 2, 'n_estimators': 96, 'learning_rate': 0.18351582496192487, 'num_leaves': 206, 'min_child_samples': 3}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:21,226][0m Trial 216 finished with value: 48229.27523702702 and parameters: {'max_depth': 6, 'n_estimators': 93, 'learning_rate': 0.23040102571209353, 'num_leaves': 720, 'min_child_samples': 13}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:21,315][0m Trial 217 finished with value: 48601.40781835426 and parameters: {'max_depth': 6, 'n_estimators': 97, 'learning_rate': 0.15628489221036038, 'num_leaves': 482, 'min_child_samples': 24}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:21,416][0m Trial 218 finished with value: 48034.57821389065 and parameters: {'max_depth': 6, 'n_estimators': 100, 'learning_rate': 0.2064624961443968, 'num_leaves': 901, 'min_child_samples': 6}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:21,510][0m Trial 219 finished with value: 48235.527544819925 and parameters: {'max_depth': 6, 'n_estimators': 99, 'learning_rate': 0.21054094324856365, 'num_leaves': 304, 'min_child_samples': 9}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:21,594][0m Trial 220 finished with value: 48563.30436725415 and parameters: {'max_depth': 6, 'n_estimators': 100, 'learning_rate': 0.2483661002000759, 'num_leaves': 575, 'min_child_samples': 17}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:21,690][0m Trial 221 finished with value: 47915.047306051165 and parameters: {'max_depth': 6, 'n_estimators': 98, 'learning_rate': 0.19288668016145497, 'num_leaves': 893, 'min_child_samples': 6}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:21,792][0m Trial 222 finished with value: 48121.79846624286 and parameters: {'max_depth': 6, 'n_estimators': 98, 'learning_rate': 0.20240201043906506, 'num_leaves': 1000, 'min_child_samples': 3}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:21,896][0m Trial 223 finished with value: 48419.448012577835 and parameters: {'max_depth': 6, 'n_estimators': 100, 'learning_rate': 0.19932089703774786, 'num_leaves': 1053, 'min_child_samples': 3}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:21,999][0m Trial 224 finished with value: 47877.47572201632 and parameters: {'max_depth': 6, 'n_estimators': 98, 'learning_rate': 0.2163546036153633, 'num_leaves': 939, 'min_child_samples': 3}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:22,100][0m Trial 225 finished with value: 48497.368282936106 and parameters: {'max_depth': 6, 'n_estimators': 98, 'learning_rate': 0.21856734389601115, 'num_leaves': 939, 'min_child_samples': 3}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:22,193][0m Trial 226 finished with value: 48032.265829853044 and parameters: {'max_depth': 6, 'n_estimators': 97, 'learning_rate': 0.2322621250881164, 'num_leaves': 1070, 'min_child_samples': 12}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:22,284][0m Trial 227 finished with value: 48342.738782938286 and parameters: {'max_depth': 6, 'n_estimators': 96, 'learning_rate': 0.23888224684779377, 'num_leaves': 1138, 'min_child_samples': 13}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:22,379][0m Trial 228 finished with value: 48411.448858138596 and parameters: {'max_depth': 6, 'n_estimators': 97, 'learning_rate': 0.22141694926411631, 'num_leaves': 1055, 'min_child_samples': 11}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:22,466][0m Trial 229 finished with value: 48613.74403855188 and parameters: {'max_depth': 6, 'n_estimators': 100, 'learning_rate': 0.2575529681386528, 'num_leaves': 873, 'min_child_samples': 20}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:22,549][0m Trial 230 finished with value: 48443.354789265264 and parameters: {'max_depth': 6, 'n_estimators': 95, 'learning_rate': 0.2755855109313287, 'num_leaves': 892, 'min_child_samples': 15}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:22,649][0m Trial 231 finished with value: 47686.37116203756 and parameters: {'max_depth': 6, 'n_estimators': 98, 'learning_rate': 0.1990608573338187, 'num_leaves': 1065, 'min_child_samples': 6}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:22,742][0m Trial 232 finished with value: 48134.75238166177 and parameters: {'max_depth': 6, 'n_estimators': 98, 'learning_rate': 0.23088486551815285, 'num_leaves': 1109, 'min_child_samples': 8}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:22,836][0m Trial 233 finished with value: 47983.280949585016 and parameters: {'max_depth': 6, 'n_estimators': 97, 'learning_rate': 0.18557630831030114, 'num_leaves': 1212, 'min_child_samples': 11}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:22,929][0m Trial 234 finished with value: 48138.683474465106 and parameters: {'max_depth': 6, 'n_estimators': 96, 'learning_rate': 0.1880973915246, 'num_leaves': 1229, 'min_child_samples': 15}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:23,032][0m Trial 235 finished with value: 48233.90456535716 and parameters: {'max_depth': 6, 'n_estimators': 100, 'learning_rate': 0.18491649371680477, 'num_leaves': 1187, 'min_child_samples': 6}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:23,122][0m Trial 236 finished with value: 48806.8202754537 and parameters: {'max_depth': 6, 'n_estimators': 95, 'learning_rate': 0.21755360625314066, 'num_leaves': 1004, 'min_child_samples': 12}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:23,224][0m Trial 237 finished with value: 49318.83915964459 and parameters: {'max_depth': 6, 'n_estimators': 97, 'learning_rate': 0.29407860204280784, 'num_leaves': 982, 'min_child_samples': 3}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:23,311][0m Trial 238 finished with value: 48538.530531122204 and parameters: {'max_depth': 6, 'n_estimators': 98, 'learning_rate': 0.24769190610393008, 'num_leaves': 894, 'min_child_samples': 18}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:23,411][0m Trial 239 finished with value: 48118.255385331024 and parameters: {'max_depth': 6, 'n_estimators': 100, 'learning_rate': 0.1872986601987921, 'num_leaves': 1311, 'min_child_samples': 6}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:23,506][0m Trial 240 finished with value: 48112.259997653906 and parameters: {'max_depth': 6, 'n_estimators': 92, 'learning_rate': 0.20836448926173803, 'num_leaves': 786, 'min_child_samples': 10}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:23,597][0m Trial 241 finished with value: 48506.74075682804 and parameters: {'max_depth': 6, 'n_estimators': 91, 'learning_rate': 0.2023899947808585, 'num_leaves': 833, 'min_child_samples': 10}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:23,685][0m Trial 242 finished with value: 48339.617015025564 and parameters: {'max_depth': 6, 'n_estimators': 94, 'learning_rate': 0.22365718475737936, 'num_leaves': 752, 'min_child_samples': 13}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:23,781][0m Trial 243 finished with value: 48179.51260760516 and parameters: {'max_depth': 6, 'n_estimators': 92, 'learning_rate': 0.18061168005741676, 'num_leaves': 2720, 'min_child_samples': 7}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:23,869][0m Trial 244 finished with value: 48614.44473567351 and parameters: {'max_depth': 6, 'n_estimators': 95, 'learning_rate': 0.20727790320291226, 'num_leaves': 4956, 'min_child_samples': 16}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:23,973][0m Trial 245 finished with value: 48452.926853257566 and parameters: {'max_depth': 6, 'n_estimators': 97, 'learning_rate': 0.23418334817437203, 'num_leaves': 1058, 'min_child_samples': 3}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:24,064][0m Trial 246 finished with value: 48769.92517838087 and parameters: {'max_depth': 6, 'n_estimators': 90, 'learning_rate': 0.17299146679535654, 'num_leaves': 915, 'min_child_samples': 10}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:24,153][0m Trial 247 finished with value: 48504.05070894635 and parameters: {'max_depth': 6, 'n_estimators': 100, 'learning_rate': 0.19535977829512252, 'num_leaves': 764, 'min_child_samples': 21}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:24,246][0m Trial 248 finished with value: 51651.96584450165 and parameters: {'max_depth': 6, 'n_estimators': 93, 'learning_rate': 0.6820002799256016, 'num_leaves': 1120, 'min_child_samples': 6}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:24,333][0m Trial 249 finished with value: 48305.32851045701 and parameters: {'max_depth': 6, 'n_estimators': 96, 'learning_rate': 0.25670688676676134, 'num_leaves': 970, 'min_child_samples': 13}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:24,426][0m Trial 250 finished with value: 48412.494840357474 and parameters: {'max_depth': 6, 'n_estimators': 98, 'learning_rate': 0.20749281854269833, 'num_leaves': 757, 'min_child_samples': 10}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:24,501][0m Trial 251 finished with value: 49034.29304752203 and parameters: {'max_depth': 6, 'n_estimators': 88, 'learning_rate': 0.23222839248823704, 'num_leaves': 158, 'min_child_samples': 95}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:24,539][0m Trial 252 finished with value: 51769.50706703394 and parameters: {'max_depth': 6, 'n_estimators': 10, 'learning_rate': 0.9264611216632098, 'num_leaves': 893, 'min_child_samples': 17}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:24,641][0m Trial 253 finished with value: 48227.49595249695 and parameters: {'max_depth': 6, 'n_estimators': 95, 'learning_rate': 0.174366896254597, 'num_leaves': 826, 'min_child_samples': 3}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:24,731][0m Trial 254 finished with value: 48140.507673250475 and parameters: {'max_depth': 6, 'n_estimators': 100, 'learning_rate': 0.21395581349348694, 'num_leaves': 696, 'min_child_samples': 8}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:24,817][0m Trial 255 finished with value: 48665.33371639314 and parameters: {'max_depth': 6, 'n_estimators': 98, 'learning_rate': 0.31732431538988104, 'num_leaves': 221, 'min_child_samples': 24}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:24,932][0m Trial 256 finished with value: 48128.81688670178 and parameters: {'max_depth': 6, 'n_estimators': 92, 'learning_rate': 0.19097255924220957, 'num_leaves': 102, 'min_child_samples': 14}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:25,024][0m Trial 257 finished with value: 48741.12038242126 and parameters: {'max_depth': 6, 'n_estimators': 94, 'learning_rate': 0.27924382321999314, 'num_leaves': 969, 'min_child_samples': 6}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:25,117][0m Trial 258 finished with value: 48374.71837434546 and parameters: {'max_depth': 6, 'n_estimators': 96, 'learning_rate': 0.15150741904068724, 'num_leaves': 1198, 'min_child_samples': 19}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:25,223][0m Trial 259 finished with value: 48589.58735360189 and parameters: {'max_depth': 6, 'n_estimators': 100, 'learning_rate': 0.24906698090934215, 'num_leaves': 1832, 'min_child_samples': 3}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:25,315][0m Trial 260 finished with value: 49255.27001287967 and parameters: {'max_depth': 6, 'n_estimators': 98, 'learning_rate': 0.4345388769685632, 'num_leaves': 295, 'min_child_samples': 11}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:25,405][0m Trial 261 finished with value: 48332.84080724723 and parameters: {'max_depth': 6, 'n_estimators': 90, 'learning_rate': 0.16946864184788166, 'num_leaves': 692, 'min_child_samples': 16}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:25,499][0m Trial 262 finished with value: 47917.421477835545 and parameters: {'max_depth': 6, 'n_estimators': 96, 'learning_rate': 0.2221373916989625, 'num_leaves': 533, 'min_child_samples': 9}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:25,605][0m Trial 263 finished with value: 48011.349781125864 and parameters: {'max_depth': 6, 'n_estimators': 96, 'learning_rate': 0.23232688628244555, 'num_leaves': 486, 'min_child_samples': 3}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:25,709][0m Trial 264 finished with value: 48808.04093872896 and parameters: {'max_depth': 6, 'n_estimators': 96, 'learning_rate': 0.24634974601957588, 'num_leaves': 502, 'min_child_samples': 3}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:25,785][0m Trial 265 finished with value: 48750.71423979186 and parameters: {'max_depth': 6, 'n_estimators': 67, 'learning_rate': 0.27323451254972475, 'num_leaves': 456, 'min_child_samples': 6}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:25,881][0m Trial 266 finished with value: 48219.6152944042 and parameters: {'max_depth': 6, 'n_estimators': 94, 'learning_rate': 0.2303733904371138, 'num_leaves': 385, 'min_child_samples': 6}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:25,964][0m Trial 267 finished with value: 48873.49799949733 and parameters: {'max_depth': 6, 'n_estimators': 100, 'learning_rate': 0.19315072410634987, 'num_leaves': 512, 'min_child_samples': 77}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:26,065][0m Trial 268 finished with value: 48782.00695629642 and parameters: {'max_depth': 6, 'n_estimators': 96, 'learning_rate': 0.22382616700962715, 'num_leaves': 284, 'min_child_samples': 3}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:26,161][0m Trial 269 finished with value: 48094.96171202582 and parameters: {'max_depth': 6, 'n_estimators': 98, 'learning_rate': 0.17939156537205728, 'num_leaves': 581, 'min_child_samples': 8}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:26,267][0m Trial 270 finished with value: 48526.90532605394 and parameters: {'max_depth': 6, 'n_estimators': 98, 'learning_rate': 0.16152732127001448, 'num_leaves': 552, 'min_child_samples': 3}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:26,372][0m Trial 271 finished with value: 48043.83204263295 and parameters: {'max_depth': 6, 'n_estimators': 100, 'learning_rate': 0.12862668743785916, 'num_leaves': 567, 'min_child_samples': 7}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:26,475][0m Trial 272 finished with value: 48357.36924311621 and parameters: {'max_depth': 6, 'n_estimators': 100, 'learning_rate': 0.13097529483249637, 'num_leaves': 619, 'min_child_samples': 8}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:26,586][0m Trial 273 finished with value: 48401.65072144481 and parameters: {'max_depth': 6, 'n_estimators': 100, 'learning_rate': 0.10939362216619314, 'num_leaves': 599, 'min_child_samples': 3}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:26,687][0m Trial 274 finished with value: 48031.388169756516 and parameters: {'max_depth': 6, 'n_estimators': 98, 'learning_rate': 0.15389561893709355, 'num_leaves': 522, 'min_child_samples': 8}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:26,791][0m Trial 275 finished with value: 48290.71008268361 and parameters: {'max_depth': 6, 'n_estimators': 100, 'learning_rate': 0.14123526621215862, 'num_leaves': 422, 'min_child_samples': 6}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:26,896][0m Trial 276 finished with value: 48290.213280703334 and parameters: {'max_depth': 6, 'n_estimators': 97, 'learning_rate': 0.129489996679614, 'num_leaves': 496, 'min_child_samples': 3}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:26,994][0m Trial 277 finished with value: 48546.64095614478 and parameters: {'max_depth': 6, 'n_estimators': 98, 'learning_rate': 0.15122207424140366, 'num_leaves': 725, 'min_child_samples': 11}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:27,066][0m Trial 278 finished with value: 50587.949562015376 and parameters: {'max_depth': 6, 'n_estimators': 95, 'learning_rate': 0.11387075344119035, 'num_leaves': 425, 'min_child_samples': 191}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:27,165][0m Trial 279 finished with value: 48357.955731162154 and parameters: {'max_depth': 6, 'n_estimators': 100, 'learning_rate': 0.16213361191446507, 'num_leaves': 643, 'min_child_samples': 9}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:27,264][0m Trial 280 finished with value: 47866.8016714757 and parameters: {'max_depth': 6, 'n_estimators': 97, 'learning_rate': 0.18013065543252219, 'num_leaves': 536, 'min_child_samples': 8}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:27,372][0m Trial 281 finished with value: 48698.88638527306 and parameters: {'max_depth': 6, 'n_estimators': 96, 'learning_rate': 0.09389291053512344, 'num_leaves': 535, 'min_child_samples': 6}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:27,472][0m Trial 282 finished with value: 53121.38120024648 and parameters: {'max_depth': 6, 'n_estimators': 94, 'learning_rate': 0.8099057941564946, 'num_leaves': 375, 'min_child_samples': 3}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:27,538][0m Trial 283 finished with value: 48806.15844169798 and parameters: {'max_depth': 6, 'n_estimators': 46, 'learning_rate': 0.2639545413353793, 'num_leaves': 475, 'min_child_samples': 3}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:27,634][0m Trial 284 finished with value: 48688.47342058752 and parameters: {'max_depth': 6, 'n_estimators': 98, 'learning_rate': 0.1456920240638434, 'num_leaves': 678, 'min_child_samples': 10}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:27,734][0m Trial 285 finished with value: 47766.02064995536 and parameters: {'max_depth': 6, 'n_estimators': 100, 'learning_rate': 0.1751988288141121, 'num_leaves': 1106, 'min_child_samples': 7}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:27,836][0m Trial 286 finished with value: 49115.864404673266 and parameters: {'max_depth': 6, 'n_estimators': 100, 'learning_rate': 0.3022122881050161, 'num_leaves': 1280, 'min_child_samples': 3}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:27,932][0m Trial 287 finished with value: 48206.12229320484 and parameters: {'max_depth': 6, 'n_estimators': 100, 'learning_rate': 0.1814268425243686, 'num_leaves': 1102, 'min_child_samples': 11}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:28,026][0m Trial 288 finished with value: 49352.785112553196 and parameters: {'max_depth': 6, 'n_estimators': 98, 'learning_rate': 0.34395733180442933, 'num_leaves': 1051, 'min_child_samples': 8}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:28,090][0m Trial 289 finished with value: 49316.6150733174 and parameters: {'max_depth': 4, 'n_estimators': 100, 'learning_rate': 0.2433974462942811, 'num_leaves': 1217, 'min_child_samples': 6}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:28,186][0m Trial 290 finished with value: 47927.81645529258 and parameters: {'max_depth': 6, 'n_estimators': 97, 'learning_rate': 0.18726708434185918, 'num_leaves': 360, 'min_child_samples': 12}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:28,279][0m Trial 291 finished with value: 48386.1541907178 and parameters: {'max_depth': 6, 'n_estimators': 97, 'learning_rate': 0.18840547057423768, 'num_leaves': 1398, 'min_child_samples': 12}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:28,382][0m Trial 292 finished with value: 48391.28069328185 and parameters: {'max_depth': 6, 'n_estimators': 96, 'learning_rate': 0.1654633230540232, 'num_leaves': 349, 'min_child_samples': 3}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:28,484][0m Trial 293 finished with value: 48038.65890626518 and parameters: {'max_depth': 6, 'n_estimators': 99, 'learning_rate': 0.12475694888368256, 'num_leaves': 333, 'min_child_samples': 8}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:28,586][0m Trial 294 finished with value: 48659.88807366755 and parameters: {'max_depth': 6, 'n_estimators': 100, 'learning_rate': 0.13073395442648725, 'num_leaves': 415, 'min_child_samples': 12}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:28,689][0m Trial 295 finished with value: 48005.21566498076 and parameters: {'max_depth': 6, 'n_estimators': 99, 'learning_rate': 0.14816112274678395, 'num_leaves': 354, 'min_child_samples': 7}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:28,791][0m Trial 296 finished with value: 48426.24944616035 and parameters: {'max_depth': 6, 'n_estimators': 99, 'learning_rate': 0.11865960807643688, 'num_leaves': 335, 'min_child_samples': 7}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:28,902][0m Trial 297 finished with value: 48637.148223443626 and parameters: {'max_depth': 6, 'n_estimators': 100, 'learning_rate': 0.08998346113562497, 'num_leaves': 1144, 'min_child_samples': 8}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:29,011][0m Trial 298 finished with value: 48635.423871171384 and parameters: {'max_depth': 6, 'n_estimators': 98, 'learning_rate': 0.14654630234777635, 'num_leaves': 454, 'min_child_samples': 3}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:29,115][0m Trial 299 finished with value: 48396.71346227497 and parameters: {'max_depth': 6, 'n_estimators': 100, 'learning_rate': 0.11668395176979483, 'num_leaves': 1047, 'min_child_samples': 9}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:29,217][0m Trial 300 finished with value: 48546.00540053105 and parameters: {'max_depth': 6, 'n_estimators': 98, 'learning_rate': 0.15330849612569203, 'num_leaves': 4214, 'min_child_samples': 13}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:29,264][0m Trial 301 finished with value: 53038.950803983316 and parameters: {'max_depth': 6, 'n_estimators': 96, 'learning_rate': 0.17391633599551815, 'num_leaves': 3, 'min_child_samples': 6}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:29,365][0m Trial 302 finished with value: 48601.385906370815 and parameters: {'max_depth': 6, 'n_estimators': 100, 'learning_rate': 0.1310333000022002, 'num_leaves': 173, 'min_child_samples': 11}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:29,472][0m Trial 303 finished with value: 48601.104116699964 and parameters: {'max_depth': 6, 'n_estimators': 98, 'learning_rate': 0.17728503316306427, 'num_leaves': 539, 'min_child_samples': 3}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:29,572][0m Trial 304 finished with value: 48565.26712720935 and parameters: {'max_depth': 6, 'n_estimators': 96, 'learning_rate': 0.154273636055416, 'num_leaves': 310, 'min_child_samples': 7}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:29,667][0m Trial 305 finished with value: 47782.64494398311 and parameters: {'max_depth': 6, 'n_estimators': 100, 'learning_rate': 0.1974062370692603, 'num_leaves': 531, 'min_child_samples': 14}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:29,775][0m Trial 306 finished with value: 51798.795741028305 and parameters: {'max_depth': 6, 'n_estimators': 100, 'learning_rate': 0.620679387995573, 'num_leaves': 518, 'min_child_samples': 3}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:29,874][0m Trial 307 finished with value: 47854.768900365234 and parameters: {'max_depth': 6, 'n_estimators': 100, 'learning_rate': 0.18743605981925276, 'num_leaves': 549, 'min_child_samples': 10}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:29,974][0m Trial 308 finished with value: 48007.34729105424 and parameters: {'max_depth': 6, 'n_estimators': 100, 'learning_rate': 0.1942005880443749, 'num_leaves': 557, 'min_child_samples': 8}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:30,072][0m Trial 309 finished with value: 47946.07931606004 and parameters: {'max_depth': 6, 'n_estimators': 100, 'learning_rate': 0.19650886872686707, 'num_leaves': 411, 'min_child_samples': 11}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:30,166][0m Trial 310 finished with value: 48398.21954368876 and parameters: {'max_depth': 6, 'n_estimators': 100, 'learning_rate': 0.19400129641583336, 'num_leaves': 445, 'min_child_samples': 13}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:30,265][0m Trial 311 finished with value: 48356.885311030775 and parameters: {'max_depth': 6, 'n_estimators': 98, 'learning_rate': 0.22035406107427186, 'num_leaves': 579, 'min_child_samples': 10}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:30,374][0m Trial 312 finished with value: 48334.20880552368 and parameters: {'max_depth': 6, 'n_estimators': 100, 'learning_rate': 0.1982003678047324, 'num_leaves': 487, 'min_child_samples': 3}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:30,474][0m Trial 313 finished with value: 48746.13369125733 and parameters: {'max_depth': 6, 'n_estimators': 97, 'learning_rate': 0.19742896723692005, 'num_leaves': 642, 'min_child_samples': 13}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:30,580][0m Trial 314 finished with value: 48613.75026645628 and parameters: {'max_depth': 6, 'n_estimators': 98, 'learning_rate': 0.22171024492311628, 'num_leaves': 412, 'min_child_samples': 6}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:30,686][0m Trial 315 finished with value: 48005.983336907724 and parameters: {'max_depth': 6, 'n_estimators': 100, 'learning_rate': 0.17659425534418793, 'num_leaves': 552, 'min_child_samples': 10}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:30,784][0m Trial 316 finished with value: 47974.734873393754 and parameters: {'max_depth': 6, 'n_estimators': 100, 'learning_rate': 0.17442854977369493, 'num_leaves': 531, 'min_child_samples': 15}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:30,880][0m Trial 317 finished with value: 48588.61044513592 and parameters: {'max_depth': 6, 'n_estimators': 100, 'learning_rate': 0.16868983070217813, 'num_leaves': 527, 'min_child_samples': 16}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:30,979][0m Trial 318 finished with value: 48107.715360040274 and parameters: {'max_depth': 6, 'n_estimators': 98, 'learning_rate': 0.1689039143692229, 'num_leaves': 428, 'min_child_samples': 16}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:31,084][0m Trial 319 finished with value: 48374.51459783386 and parameters: {'max_depth': 6, 'n_estimators': 100, 'learning_rate': 0.18190818891832106, 'num_leaves': 539, 'min_child_samples': 11}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:31,177][0m Trial 320 finished with value: 48310.04354432328 and parameters: {'max_depth': 6, 'n_estimators': 96, 'learning_rate': 0.15547794275598198, 'num_leaves': 391, 'min_child_samples': 19}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:31,282][0m Trial 321 finished with value: 47745.87040134311 and parameters: {'max_depth': 6, 'n_estimators': 100, 'learning_rate': 0.18028530394051043, 'num_leaves': 270, 'min_child_samples': 13}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:31,366][0m Trial 322 finished with value: 49237.44521067663 and parameters: {'max_depth': 6, 'n_estimators': 100, 'learning_rate': 0.1785572211498398, 'num_leaves': 275, 'min_child_samples': 85}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:31,467][0m Trial 323 finished with value: 48659.49746234774 and parameters: {'max_depth': 6, 'n_estimators': 100, 'learning_rate': 0.159647916085017, 'num_leaves': 346, 'min_child_samples': 14}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:31,552][0m Trial 324 finished with value: 48659.25312047023 and parameters: {'max_depth': 6, 'n_estimators': 98, 'learning_rate': 0.18691778071755916, 'num_leaves': 208, 'min_child_samples': 67}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:31,655][0m Trial 325 finished with value: 48113.7377306434 and parameters: {'max_depth': 6, 'n_estimators': 100, 'learning_rate': 0.1976160614647415, 'num_leaves': 461, 'min_child_samples': 10}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:31,713][0m Trial 326 finished with value: 50804.73724717054 and parameters: {'max_depth': 3, 'n_estimators': 98, 'learning_rate': 0.1472728770923794, 'num_leaves': 622, 'min_child_samples': 20}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:31,818][0m Trial 327 finished with value: 48457.20312463235 and parameters: {'max_depth': 6, 'n_estimators': 100, 'learning_rate': 0.1683445389397896, 'num_leaves': 288, 'min_child_samples': 15}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:31,912][0m Trial 328 finished with value: 48311.78238755595 and parameters: {'max_depth': 6, 'n_estimators': 96, 'learning_rate': 0.19524072845751608, 'num_leaves': 540, 'min_child_samples': 10}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:31,991][0m Trial 329 finished with value: 49469.4270171537 and parameters: {'max_depth': 6, 'n_estimators': 98, 'learning_rate': 0.17616643032963222, 'num_leaves': 384, 'min_child_samples': 122}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:32,089][0m Trial 330 finished with value: 48176.90124987669 and parameters: {'max_depth': 6, 'n_estimators': 100, 'learning_rate': 0.21533036976388636, 'num_leaves': 501, 'min_child_samples': 9}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:32,185][0m Trial 331 finished with value: 48052.39277966747 and parameters: {'max_depth': 6, 'n_estimators': 96, 'learning_rate': 0.1456093415041903, 'num_leaves': 625, 'min_child_samples': 16}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:32,298][0m Trial 332 finished with value: 48038.91458907256 and parameters: {'max_depth': 6, 'n_estimators': 98, 'learning_rate': 0.1883339693826763, 'num_leaves': 259, 'min_child_samples': 3}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:32,404][0m Trial 333 finished with value: 48169.934615055274 and parameters: {'max_depth': 6, 'n_estimators': 100, 'learning_rate': 0.16660721173962004, 'num_leaves': 2000, 'min_child_samples': 7}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:32,497][0m Trial 334 finished with value: 48347.92725877268 and parameters: {'max_depth': 6, 'n_estimators': 95, 'learning_rate': 0.2095237713917332, 'num_leaves': 713, 'min_child_samples': 13}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:32,593][0m Trial 335 finished with value: 47923.59194545384 and parameters: {'max_depth': 6, 'n_estimators': 98, 'learning_rate': 0.23043304698017714, 'num_leaves': 399, 'min_child_samples': 10}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:32,686][0m Trial 336 finished with value: 48616.15271168911 and parameters: {'max_depth': 6, 'n_estimators': 100, 'learning_rate': 0.23645318838475105, 'num_leaves': 359, 'min_child_samples': 20}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:32,744][0m Trial 337 finished with value: 49461.99456865944 and parameters: {'max_depth': 6, 'n_estimators': 30, 'learning_rate': 0.22125836522520065, 'num_leaves': 169, 'min_child_samples': 11}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:32,827][0m Trial 338 finished with value: 48653.25335271383 and parameters: {'max_depth': 6, 'n_estimators': 76, 'learning_rate': 0.19889874131086197, 'num_leaves': 415, 'min_child_samples': 16}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:32,935][0m Trial 339 finished with value: 48289.27003867731 and parameters: {'max_depth': 6, 'n_estimators': 96, 'learning_rate': 0.22863259845453482, 'num_leaves': 268, 'min_child_samples': 3}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:33,031][0m Trial 340 finished with value: 48240.635194643466 and parameters: {'max_depth': 6, 'n_estimators': 98, 'learning_rate': 0.2143191017116878, 'num_leaves': 458, 'min_child_samples': 6}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:33,132][0m Trial 341 finished with value: 47644.76605941187 and parameters: {'max_depth': 6, 'n_estimators': 100, 'learning_rate': 0.1875949715392314, 'num_leaves': 607, 'min_child_samples': 14}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:33,221][0m Trial 342 finished with value: 48534.55589707889 and parameters: {'max_depth': 6, 'n_estimators': 95, 'learning_rate': 0.24220042131461839, 'num_leaves': 369, 'min_child_samples': 25}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:33,315][0m Trial 343 finished with value: 48525.32572272776 and parameters: {'max_depth': 6, 'n_estimators': 100, 'learning_rate': 0.19614522186818617, 'num_leaves': 588, 'min_child_samples': 21}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:33,418][0m Trial 344 finished with value: 48079.22319329354 and parameters: {'max_depth': 6, 'n_estimators': 98, 'learning_rate': 0.17966295250669287, 'num_leaves': 76, 'min_child_samples': 15}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:33,521][0m Trial 345 finished with value: 48355.21467508716 and parameters: {'max_depth': 6, 'n_estimators': 100, 'learning_rate': 0.2046723669684132, 'num_leaves': 186, 'min_child_samples': 13}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:33,612][0m Trial 346 finished with value: 48368.88722305093 and parameters: {'max_depth': 6, 'n_estimators': 97, 'learning_rate': 0.37492463607724585, 'num_leaves': 456, 'min_child_samples': 18}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:33,688][0m Trial 347 finished with value: 48933.21559675687 and parameters: {'max_depth': 5, 'n_estimators': 95, 'learning_rate': 0.23056361649027912, 'num_leaves': 313, 'min_child_samples': 10}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:33,789][0m Trial 348 finished with value: 48155.3712521118 and parameters: {'max_depth': 6, 'n_estimators': 100, 'learning_rate': 0.1814027914804861, 'num_leaves': 706, 'min_child_samples': 14}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:33,885][0m Trial 349 finished with value: 49922.38441430326 and parameters: {'max_depth': 6, 'n_estimators': 97, 'learning_rate': 0.5147480825833148, 'num_leaves': 3354, 'min_child_samples': 9}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:33,978][0m Trial 350 finished with value: 48424.9452482572 and parameters: {'max_depth': 6, 'n_estimators': 94, 'learning_rate': 0.2151886327400182, 'num_leaves': 558, 'min_child_samples': 18}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:34,084][0m Trial 351 finished with value: 48018.70621121422 and parameters: {'max_depth': 6, 'n_estimators': 98, 'learning_rate': 0.1930609549383287, 'num_leaves': 424, 'min_child_samples': 12}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:34,193][0m Trial 352 finished with value: 48389.951348032446 and parameters: {'max_depth': 6, 'n_estimators': 100, 'learning_rate': 0.25343530377831325, 'num_leaves': 250, 'min_child_samples': 7}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:34,291][0m Trial 353 finished with value: 48574.597738795244 and parameters: {'max_depth': 6, 'n_estimators': 96, 'learning_rate': 0.1670103684931605, 'num_leaves': 777, 'min_child_samples': 22}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:34,400][0m Trial 354 finished with value: 48480.077693502324 and parameters: {'max_depth': 6, 'n_estimators': 100, 'learning_rate': 0.1436880453770513, 'num_leaves': 627, 'min_child_samples': 11}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:34,476][0m Trial 355 finished with value: 49378.81607951799 and parameters: {'max_depth': 6, 'n_estimators': 98, 'learning_rate': 0.20377017878151665, 'num_leaves': 513, 'min_child_samples': 152}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:34,585][0m Trial 356 finished with value: 48707.96634024011 and parameters: {'max_depth': 6, 'n_estimators': 95, 'learning_rate': 0.2377075269354183, 'num_leaves': 1604, 'min_child_samples': 3}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:34,695][0m Trial 357 finished with value: 48119.72919283328 and parameters: {'max_depth': 6, 'n_estimators': 100, 'learning_rate': 0.17718919919596976, 'num_leaves': 325, 'min_child_samples': 6}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:34,788][0m Trial 358 finished with value: 48520.8532038592 and parameters: {'max_depth': 6, 'n_estimators': 98, 'learning_rate': 0.22038653173515768, 'num_leaves': 436, 'min_child_samples': 16}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:34,871][0m Trial 359 finished with value: 48921.169156570446 and parameters: {'max_depth': 6, 'n_estimators': 62, 'learning_rate': 0.15982728470811058, 'num_leaves': 119, 'min_child_samples': 10}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:34,995][0m Trial 360 finished with value: 48571.02681972639 and parameters: {'max_depth': 6, 'n_estimators': 96, 'learning_rate': 0.19263693872333515, 'num_leaves': 672, 'min_child_samples': 14}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:35,099][0m Trial 361 finished with value: 47552.620852375316 and parameters: {'max_depth': 6, 'n_estimators': 100, 'learning_rate': 0.21375264143110428, 'num_leaves': 585, 'min_child_samples': 6}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:35,196][0m Trial 362 finished with value: 48464.962607199996 and parameters: {'max_depth': 6, 'n_estimators': 100, 'learning_rate': 0.1767573617433532, 'num_leaves': 810, 'min_child_samples': 18}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:35,316][0m Trial 363 finished with value: 48052.19898775122 and parameters: {'max_depth': 6, 'n_estimators': 100, 'learning_rate': 0.2070682703621044, 'num_leaves': 585, 'min_child_samples': 7}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:35,424][0m Trial 364 finished with value: 48470.526671872685 and parameters: {'max_depth': 6, 'n_estimators': 98, 'learning_rate': 0.13987825725052572, 'num_leaves': 711, 'min_child_samples': 12}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:35,536][0m Trial 365 finished with value: 47918.229226117444 and parameters: {'max_depth': 6, 'n_estimators': 100, 'learning_rate': 0.16226193368770572, 'num_leaves': 588, 'min_child_samples': 9}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:35,633][0m Trial 366 finished with value: 48324.43559120771 and parameters: {'max_depth': 6, 'n_estimators': 98, 'learning_rate': 0.15830184157740632, 'num_leaves': 356, 'min_child_samples': 22}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:35,738][0m Trial 367 finished with value: 48385.70664026169 and parameters: {'max_depth': 6, 'n_estimators': 100, 'learning_rate': 0.1388597972844075, 'num_leaves': 753, 'min_child_samples': 13}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:35,871][0m Trial 368 finished with value: 99756.40394589196 and parameters: {'max_depth': 6, 'n_estimators': 98, 'learning_rate': 0.0004923291274416264, 'num_leaves': 189, 'min_child_samples': 16}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:35,989][0m Trial 369 finished with value: 48471.035534697585 and parameters: {'max_depth': 6, 'n_estimators': 100, 'learning_rate': 0.16179924019408226, 'num_leaves': 610, 'min_child_samples': 3}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:36,098][0m Trial 370 finished with value: 48368.78299011855 and parameters: {'max_depth': 6, 'n_estimators': 94, 'learning_rate': 0.10323764070915424, 'num_leaves': 471, 'min_child_samples': 9}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:36,205][0m Trial 371 finished with value: 48481.2611083335 and parameters: {'max_depth': 6, 'n_estimators': 97, 'learning_rate': 0.1776937606865122, 'num_leaves': 279, 'min_child_samples': 6}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:36,306][0m Trial 372 finished with value: 50257.23724666807 and parameters: {'max_depth': 6, 'n_estimators': 100, 'learning_rate': 0.4748037480746877, 'num_leaves': 695, 'min_child_samples': 11}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:36,401][0m Trial 373 finished with value: 48528.2581286734 and parameters: {'max_depth': 6, 'n_estimators': 98, 'learning_rate': 0.210671045684075, 'num_leaves': 384, 'min_child_samples': 19}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:36,508][0m Trial 374 finished with value: 48566.35115618422 and parameters: {'max_depth': 6, 'n_estimators': 96, 'learning_rate': 0.13634437378640155, 'num_leaves': 842, 'min_child_samples': 6}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:36,605][0m Trial 375 finished with value: 48284.76636524252 and parameters: {'max_depth': 6, 'n_estimators': 100, 'learning_rate': 0.1808964164040392, 'num_leaves': 2353, 'min_child_samples': 14}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:36,670][0m Trial 376 finished with value: 49472.544439144076 and parameters: {'max_depth': 6, 'n_estimators': 94, 'learning_rate': 0.15749144767989376, 'num_leaves': 12, 'min_child_samples': 3}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:36,759][0m Trial 377 finished with value: 48716.89897086582 and parameters: {'max_depth': 6, 'n_estimators': 73, 'learning_rate': 0.2575098837729067, 'num_leaves': 552, 'min_child_samples': 10}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:36,853][0m Trial 378 finished with value: 48785.608205832395 and parameters: {'max_depth': 6, 'n_estimators': 98, 'learning_rate': 0.19788844896519744, 'num_leaves': 421, 'min_child_samples': 28}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:36,953][0m Trial 379 finished with value: 48489.70636642119 and parameters: {'max_depth': 6, 'n_estimators': 100, 'learning_rate': 0.22209510150509448, 'num_leaves': 509, 'min_child_samples': 16}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:37,063][0m Trial 380 finished with value: 47617.541393847954 and parameters: {'max_depth': 6, 'n_estimators': 96, 'learning_rate': 0.18607346318030515, 'num_leaves': 660, 'min_child_samples': 6}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:37,174][0m Trial 381 finished with value: 48618.44641046155 and parameters: {'max_depth': 6, 'n_estimators': 94, 'learning_rate': 0.2370133786257302, 'num_leaves': 935, 'min_child_samples': 3}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:37,264][0m Trial 382 finished with value: 48371.431524028994 and parameters: {'max_depth': 6, 'n_estimators': 96, 'learning_rate': 0.19651415629601257, 'num_leaves': 826, 'min_child_samples': 60}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:37,371][0m Trial 383 finished with value: 48394.15050709075 and parameters: {'max_depth': 6, 'n_estimators': 96, 'learning_rate': 0.21397094058750227, 'num_leaves': 663, 'min_child_samples': 6}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:37,486][0m Trial 384 finished with value: 48767.80428849173 and parameters: {'max_depth': 6, 'n_estimators': 94, 'learning_rate': 0.11381658456568042, 'num_leaves': 297, 'min_child_samples': 3}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:37,593][0m Trial 385 finished with value: 48406.201132938804 and parameters: {'max_depth': 6, 'n_estimators': 97, 'learning_rate': 0.15982690056518575, 'num_leaves': 223, 'min_child_samples': 8}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:37,699][0m Trial 386 finished with value: 48353.96939433172 and parameters: {'max_depth': 6, 'n_estimators': 98, 'learning_rate': 0.20057379243567697, 'num_leaves': 653, 'min_child_samples': 12}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:37,806][0m Trial 387 finished with value: 48180.301658597644 and parameters: {'max_depth': 6, 'n_estimators': 96, 'learning_rate': 0.17842346257255776, 'num_leaves': 768, 'min_child_samples': 6}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:37,906][0m Trial 388 finished with value: 48545.8176569222 and parameters: {'max_depth': 6, 'n_estimators': 98, 'learning_rate': 0.1402974779735017, 'num_leaves': 381, 'min_child_samples': 15}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:38,010][0m Trial 389 finished with value: 48547.388809636446 and parameters: {'max_depth': 6, 'n_estimators': 93, 'learning_rate': 0.22538881743818517, 'num_leaves': 3002, 'min_child_samples': 3}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:38,079][0m Trial 390 finished with value: 49244.85650226103 and parameters: {'max_depth': 4, 'n_estimators': 100, 'learning_rate': 0.18995969387981787, 'num_leaves': 122, 'min_child_samples': 9}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:38,148][0m Trial 391 finished with value: 49146.43099839006 and parameters: {'max_depth': 6, 'n_estimators': 52, 'learning_rate': 0.24633514331570072, 'num_leaves': 615, 'min_child_samples': 22}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:38,245][0m Trial 392 finished with value: 48234.25926297507 and parameters: {'max_depth': 6, 'n_estimators': 98, 'learning_rate': 0.15981623549387103, 'num_leaves': 473, 'min_child_samples': 18}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:38,318][0m Trial 393 finished with value: 49881.63233436249 and parameters: {'max_depth': 6, 'n_estimators': 95, 'learning_rate': 0.21074994375320608, 'num_leaves': 761, 'min_child_samples': 178}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:38,421][0m Trial 394 finished with value: 48226.06726935803 and parameters: {'max_depth': 6, 'n_estimators': 100, 'learning_rate': 0.1758282997658518, 'num_leaves': 1286, 'min_child_samples': 12}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:38,524][0m Trial 395 finished with value: 48392.292540625545 and parameters: {'max_depth': 6, 'n_estimators': 97, 'learning_rate': 0.1429140552415835, 'num_leaves': 908, 'min_child_samples': 6}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:38,591][0m Trial 396 finished with value: 49647.24889981215 and parameters: {'max_depth': 6, 'n_estimators': 40, 'learning_rate': 0.5820202970534951, 'num_leaves': 332, 'min_child_samples': 3}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:38,695][0m Trial 397 finished with value: 48352.13935213571 and parameters: {'max_depth': 6, 'n_estimators': 100, 'learning_rate': 0.12536022926226503, 'num_leaves': 227, 'min_child_samples': 10}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:38,789][0m Trial 398 finished with value: 48354.879897079714 and parameters: {'max_depth': 6, 'n_estimators': 96, 'learning_rate': 0.19316258930113683, 'num_leaves': 578, 'min_child_samples': 14}. Best is trial 88 with value: 47456.40122651236.[0m
    [32m[I 2020-12-03 11:18:38,876][0m Trial 399 finished with value: 48753.743986224195 and parameters: {'max_depth': 6, 'n_estimators': 79, 'learning_rate': 0.26372786854920294, 'num_leaves': 990, 'min_child_samples': 7}. Best is trial 88 with value: 47456.40122651236.[0m
    


```python
lgb_params = study.best_params
lgb_params['random_state'] = 123
lgb = LGBMRegressor(**lgb_params)
lgb.fit(X_val_train, y_val_train)

print(reg_report(lgb,X_val_test,y_val_test))
print(lgb_params)
```

    R2 score : 0.79
    MAE loss : 33391.78
    RMSE loss : 47456.40
    error % : 0.20
    None
    {'max_depth': 6, 'n_estimators': 100, 'learning_rate': 0.22222685669158865, 'num_leaves': 325, 'min_child_samples': 4, 'random_state': 123}
    

### Data Postprocessing


```python
for i in range(1, 10):
    rfe = RFE(
        estimator=LGBMRegressor(
            learning_rate=lgb_params.get('learning_rate'),
            n_estimators=lgb_params.get('n_estimators'),
            max_depth=lgb_params.get('max_depth'),
            num_leaves=lgb_params.get('num_leaves'),
            min_child_samples=lgb_params.get('min_child_samples'),
            random_state=123
        ),
        n_features_to_select=i
    )
    pipeline = Pipeline(
        steps=[
            ('s', rfe),
            ('m', LGBMRegressor(
                learning_rate=lgb_params.get('learning_rate'),
                n_estimators=lgb_params.get('n_estimators'),
                max_depth=lgb_params.get('max_depth'),
                num_leaves=lgb_params.get('num_leaves'),
                min_child_samples=lgb_params.get('min_child_samples'),
                random_state=123
            ))
        ]
    )
    pipeline.fit(X_val_train, y_val_train)
    preds = pipeline.predict(X_val_test)

    print('Number of features: ', i)
    print(reg_report(pipeline, X_val_test, y_val_test))
```

    Number of features:  1
    R2 score : 0.25
    MAE loss : 66934.17
    RMSE loss : 89096.97
    error % : 0.38
    None
    Number of features:  2
    R2 score : 0.69
    MAE loss : 40620.99
    RMSE loss : 57561.01
    error % : 0.25
    None
    Number of features:  3
    R2 score : 0.75
    MAE loss : 36078.46
    RMSE loss : 51530.77
    error % : 0.22
    None
    Number of features:  4
    R2 score : 0.78
    MAE loss : 33649.60
    RMSE loss : 48451.59
    error % : 0.21
    None
    Number of features:  5
    R2 score : 0.78
    MAE loss : 33594.73
    RMSE loss : 48805.13
    error % : 0.21
    None
    Number of features:  6
    R2 score : 0.79
    MAE loss : 33391.78
    RMSE loss : 47456.40
    error % : 0.20
    None
    Number of features:  7
    R2 score : 0.79
    MAE loss : 33391.78
    RMSE loss : 47456.40
    error % : 0.20
    None
    Number of features:  8
    R2 score : 0.79
    MAE loss : 33391.78
    RMSE loss : 47456.40
    error % : 0.20
    None
    Number of features:  9
    R2 score : 0.79
    MAE loss : 33391.78
    RMSE loss : 47456.40
    error % : 0.20
    None
    

## Model Training
Our winner is **Gradient Boosting Regressor**. Now it's time to train that model with all of our train data to obtain the *down to earth* performance of our model.


```python
model = LGBMRegressor(
    learning_rate=lgb_params.get('learning_rate'),
    n_estimators=lgb_params.get('n_estimators'),
    max_depth=lgb_params.get('max_depth'),
    num_leaves=lgb_params.get('num_leaves'),
    min_child_samples=lgb_params.get('min_child_samples'),
    random_state=123
)
model.fit(X_train_transformed, y_train)

print(reg_report(model, X_test_transformed, y_test))
```

    R2 score : 0.79
    MAE loss : 31586.87
    RMSE loss : 46046.26
    error % : 0.20
    None
    

## Conclusion
To conclude, we created this tool to predict house pricing based on certain parameters. We achieved a RMSE of 46046.26 U$D
