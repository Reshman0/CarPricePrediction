import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Veri kümesini yükleme ve filtreleme
data = pd.read_csv('used_cars.csv')
data = data[(data['price'] > 500) & (data['price'] < 80000)]  # Anormal değerleri filtreleme

# Gerekli sütunları seçme
data = data[['price', 'year', 'manufacturer', 'fuel', 'odometer', 'transmission', 'drive', 'paint_color', 'type']]
data = data.sample(frac=0.3, random_state=42)  # Verinin %30'unu kullanarak hızlandırma

# Eksik verileri doldurma
data['year'] = data['year'].fillna(data['year'].median())
data['odometer'] = data['odometer'].fillna(data['odometer'].median())
categorical_columns = ['manufacturer', 'fuel', 'transmission', 'drive', 'paint_color', 'type']
for column in categorical_columns:
    data[column] = data[column].fillna(data[column].mode()[0])

# Kategorik sütunları sayısal değerlere dönüştürme
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
for column in categorical_columns:
    data[column] = label_encoder.fit_transform(data[column])

# Hedef ve özellikleri ayırma
X = data.drop('price', axis=1)
y = data['price']

# Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest modelini oluşturma
param_grid = {
    'n_estimators': [50, 100],  # Daha az kombinasyon
    'max_depth': [10],
    'min_samples_split': [2],
    'min_samples_leaf': [1]
}
rf_model = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=2, scoring='neg_mean_absolute_error', verbose=2)
grid_search.fit(X_train, y_train)

# En iyi modeli seçme ve tahmin yapma
best_rf_model = grid_search.best_estimator_
y_pred_rf = best_rf_model.predict(X_test)

# Performans metriklerini hesaplama
def evaluate_model(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"R² Score: {r2:.2f}")
    return mae, mse, r2

# Özellik önemlerini görselleştirme
def visualize_feature_importance(model, feature_names):
    importances = model.feature_importances_
    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, importances, color='skyblue')
    plt.xlabel("Feature Importance")
    plt.ylabel("Features")
    plt.title("Feature Importances (Random Forest)")
    plt.show()

# Hata (residual) analizini görselleştirme
def visualize_residuals(y_test, y_pred):
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=50, color='orange', edgecolor='black')
    plt.xlabel("Residuals (Errors)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Prediction Errors (Residuals)")
    plt.show()

# ---- ANA PROGRAM ---- #

# Performans metriklerini yazdır
mae, mse, r2 = evaluate_model(y_test, y_pred_rf)

# Özellik önemlerini görselleştir
visualize_feature_importance(best_rf_model, X.columns)

# Tahmin hatalarını görselleştir
visualize_residuals(y_test, y_pred_rf)
