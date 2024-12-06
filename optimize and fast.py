import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Veri Yükleme ve Filtreleme
data = pd.read_csv('used_cars.csv')
data = data[(data['price'] > 500) & (data['price'] < 80000)]  # Anormal fiyatları filtreleme
data = data[['price', 'year', 'manufacturer', 'fuel', 'odometer', 'transmission', 'drive', 'paint_color', 'type']]  # Gereksiz sütunları çıkarın
data = data.sample(frac=0.3, random_state=42)  # Verinin %30'unu kullan

# Eksik Verileri Doldurma
data['year'] = data['year'].fillna(data['year'].median()).astype('int32')
data['odometer'] = data['odometer'].fillna(data['odometer'].median()).astype('float32')

# Kategorik Değişkenleri Dönüştürme
categorical_columns = ['manufacturer', 'fuel', 'transmission', 'drive', 'paint_color', 'type']
label_encoder = LabelEncoder()
for col in categorical_columns:
    data[col] = label_encoder.fit_transform(data[col])

# Özellik ve Hedef Ayırma
X = data.drop('price', axis=1)
y = data['price']

# Veriyi Bölme
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Random Forest Modeli (Hızlı Ayarlar)
rf_model = RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42)
rf_model.fit(X_train, y_train)

# Tahmin ve Performans Değerlendirme
y_pred = rf_model.predict(X_test)

def evaluate_model(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"R² Score: {r2:.2f}")
    return mae, mse, r2

mae, mse, r2 = evaluate_model(y_test, y_pred)

# Özellik Önemini Görselleştirme
def visualize_feature_importance(model, feature_names):
    importances = model.feature_importances_
    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, importances, color='skyblue')
    plt.xlabel("Feature Importance")
    plt.ylabel("Features")
    plt.title("Feature Importances (Random Forest)")
    plt.show()

visualize_feature_importance(rf_model, X.columns)

# Tahmin ve Gerçek Fiyatları Görselleştirme
def visualize_predictions(y_test, y_pred):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2)
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.title("Actual vs Predicted Car Prices")
    plt.show()

visualize_predictions(y_test, y_pred)

# Hata (Residual) Analizini Görselleştirme
def visualize_residuals(y_test, y_pred):
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=50, color='orange', edgecolor='black')
    plt.xlabel("Residuals (Errors)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Prediction Errors (Residuals)")
    plt.show()

visualize_residuals(y_test, y_pred)
