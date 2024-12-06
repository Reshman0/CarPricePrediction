import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('used_cars.csv')

# Filter abnormal prices
data = data[(data['price'] > 500) & (data['price'] < 80000)]

# Select necessary columns
data = data[['price', 'year', 'manufacturer', 'fuel', 'odometer', 'transmission', 'drive', 'paint_color', 'type']]

# Fill missing values
data['year'] = data['year'].fillna(data['year'].median())
data['odometer'] = data['odometer'].fillna(data['odometer'].median())
categorical_columns = ['manufacturer', 'fuel', 'transmission', 'drive', 'paint_color', 'type']
for column in categorical_columns:
    data[column] = data[column].fillna(data[column].mode()[0])

# Convert categorical columns with LabelEncoder
label_encoder = LabelEncoder()
for column in categorical_columns:
    data[column] = label_encoder.fit_transform(data[column])

# Separate target and features
X = data.drop('price', axis=1)
y = data['price']

# Scale numerical features
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter optimization for Random Forest model
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

rf_model = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, scoring='neg_mean_absolute_error', verbose=2)
grid_search.fit(X_train, y_train)

# Select the best model and make predictions
best_rf_model = grid_search.best_estimator_
y_pred_rf = best_rf_model.predict(X_test)

# Performance metrics
mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
print(f"Optimized Random Forest MAE: {mae_rf}")
print(f"Optimized Random Forest MSE: {mse_rf}")

# Visualize prediction results
plt.scatter(y_test, y_pred_rf, alpha=0.5)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Car Prices (Optimized Random Forest)")
plt.show()
