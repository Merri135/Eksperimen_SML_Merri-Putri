import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def automate_diabetes_preprocessing(input_path: str, output_path: str):
    # Load dataset
    df = pd.read_csv(input_path)

    # Drop duplicates
    df.drop_duplicates(inplace=True)

    # Handle missing values
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna('Unknown')
            else:
                df[col] = df[col].fillna(df[col].median())

    # --- 📏 MinMax Scaling (hanya kolom numerik) ---
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if 'diabetes' in numeric_cols:
        numeric_cols.remove('diabetes')  # Hindari scaling target jika sudah numerik
    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # --- 🔁 Label Encoding untuk kolom kategorikal ---
    label_cols = df.select_dtypes(include='object').columns.tolist()
    for col in label_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    # Save cleaned dataset
    df.to_csv(output_path, index=False)
    print(f"✅ Preprocessing selesai. Dataset disimpan di {output_path}")

if __name__ == "__main__":
    input_path = "diabetes_prediction_dataset.csv"
    output_path = "diabetes_cleaned.csv"
    automate_diabetes_preprocessing(input_path, output_path)
