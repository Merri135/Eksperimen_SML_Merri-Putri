
# *Import Library**
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Set style for visualizations
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

# **3. Memuat Dataset**
df_pre = pd.read_csv('diabetes_prediction_dataset.csv')


# **4. Exploratory Data Analysis (EDA)**
cat_cols = ['gender', 'hypertension', 'heart_disease', 'smoking_history']
num_cols = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']


# **5. Data Preprocessing**
print('Jumlah missing value:\n', df_pre.isnull().sum())
print('Jumlah duplikat:', df_pre.duplicated().sum())

# Hapus duplikat
df_pre = df_pre.drop_duplicates()
print('Setelah bersih - duplikat:', df_pre.duplicated().sum())
print('Ukuran data:', df_pre.shape)

# Tangani missing values
for col in df_pre.columns:
    if df_pre[col].isnull().sum() > 0:
        if df_pre[col].dtype == 'object':
            df_pre[col] = df_pre[col].fillna('Unknown')
        else:
            df_pre[col] = df_pre[col].fillna(df_pre[col].median())

# MinMax Scaling numerik
scaler = MinMaxScaler()
df_scaled = df_pre.copy()
df_scaled[num_cols] = scaler.fit_transform(df_scaled[num_cols])

# Buang outlier dengan IQR
def remove_outliers_iqr(data, columns):
    df_proc = data.copy()
    for col in columns:
        Q1 = df_proc[col].quantile(0.25)
        Q3 = df_proc[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df_proc = df_proc[(df_proc[col] >= lower) & (df_proc[col] <= upper)]
    return df_proc

df_proces = remove_outliers_iqr(df_scaled, num_cols)

# Label Encoding semua kolom kategorikal
cat_features = df_proces.select_dtypes(include='object').columns.tolist()
for col in cat_features:
    le = LabelEncoder()
    df_proces[col] = le.fit_transform(df_proces[col])

# Splitting data
X = df_proces.drop(['diabetes'], axis=1)
y = df_proces['diabetes']

print("Jumlah Kolom dan baris fitur X:", X.shape)
print("Jumlah Kolom target y:", y.shape)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model: Random Forest Classifier
model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=15)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluasi classifier
print("Akurasi:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Model: Random Forest Regressor
print("\n--- Random Forest Regressor Model ---")
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train, y_train)
rf_predictions = rf_regressor.predict(X_test)
print("Random Forest R-squared:", r2_score(y_test, rf_predictions))
print("Random Forest Mean Squared Error:", mean_squared_error(y_test, rf_predictions))

# Simpan hasil preprocessing
df_proces.to_csv('diabetes_cleaned.csv', index=False)
