import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, ShuffleSplit, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from matplotlib import pyplot as plt
import matplotlib

# Configure matplotlib
matplotlib.rcParams["figure.figsize"] = (20, 10)

# Load dataset
df = pd.read_csv(r"C:\Users\Yamraj\Desktop\renvest\renvest.in\property\Bengaluru_House_Data.csv")

# Data Cleaning
df = df.drop(['area_type', 'society', 'balcony', 'availability'], axis='columns').dropna()
df['bhk'] = df['size'].apply(lambda x: int(x.split(' ')[0]))

# Convert total_sqft to numeric
def convert_sqft_to_num(x):
    try:
        if '-' in x:
            tokens = x.split('-')
            return (float(tokens[0]) + float(tokens[1])) / 2
        return float(x)
    except:
        return None

df['total_sqft'] = df['total_sqft'].apply(convert_sqft_to_num)
df = df[df['total_sqft'].notnull()]

# Add price_per_sqft feature
df['price_per_sqft'] = df['price'] * 100000 / df['total_sqft']

# Reduce dimensionality of location
df['location'] = df['location'].apply(lambda x: x.strip())
location_stats = df['location'].value_counts()
df['location'] = df['location'].apply(lambda x: 'other' if location_stats[x] <= 10 else x)

# Remove outliers
df = df[~(df['total_sqft'] / df['bhk'] < 300)]

def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        mean = np.mean(subdf['price_per_sqft'])
        std_dev = np.std(subdf['price_per_sqft'])
        reduced_df = subdf[(subdf['price_per_sqft'] > (mean - std_dev)) & (subdf['price_per_sqft'] <= (mean + std_dev))]
        df_out = pd.concat([df_out, reduced_df], ignore_index=True)
    return df_out

df = remove_pps_outliers(df)

def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df['price_per_sqft']),
                'std': np.std(bhk_df['price_per_sqft']),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk - 1)
            if stats and stats['count'] > 5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df['price_per_sqft'] < stats['mean']].index.values)
    return df.drop(exclude_indices, axis='index')

df = remove_bhk_outliers(df)
df = df[df['bath'] < df['bhk'] + 2]

# Prepare data for modeling
df = df.drop(['size', 'price_per_sqft'], axis='columns')
df = pd.get_dummies(df, drop_first=True)

X = df.drop('price', axis='columns')
y = df['price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

# Train Linear Regression model
lr_clf = LinearRegression()
lr_clf.fit(X_train, y_train)

# Evaluate model
print(f"Linear Regression Score: {lr_clf.score(X_test, y_test)}")

# Find the best model using GridSearchCV
def find_best_model_using_gridsearchcv(X, y):
    algos = {
        'linear_regression': {
            'model': LinearRegression(),
            'params': {
                'fit_intercept': [True, False],
                'copy_X': [True, False]
            }
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1, 2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion': ['squared_error', 'friedman_mse'],
                'splitter': ['best', 'random']
            }
        }
    }
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        gs = GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(X, y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })
    return pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])

print(find_best_model_using_gridsearchcv(X, y))

# Predict price
def predict_price(location, sqft, bath, bhk):
    loc_index = np.where(X.columns == location)[0][0] if location in X.columns else -1
    x = np.zeros(len(X.columns))
    x[0], x[1], x[2] = sqft, bath, bhk
    if loc_index >= 0:
        x[loc_index] = 1
    price = lr_clf.predict([x])[0]
    print(f"Price of the property is: {price}")
    return price