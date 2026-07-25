import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import mlflow
import mlflow.sklearn

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    features = ['from_grade', 'to_grade', 'action', 'implied_direction']
    target = 'accuracy_30d'
    
    df = df.dropna(subset=[target] + features)
    X = df[features].copy()
    y = df[target].copy()
    
    le = LabelEncoder()
    for col in ['from_grade', 'to_grade', 'action']:
        X[col] = le.fit_transform(X[col].astype(str))
        
    return X, y

def main():
    mlflow.set_experiment("Analyst_Rating_Prediction_CI")
    mlflow.sklearn.autolog()

    # Untuk GitHub Actions, data diletakkan di dalam repository
    data_path = os.path.join("namadataset_preprocessing", "sp_500_analyst_rating_and_price_target_accuracy.csv")
    
    try:
        X, y = load_and_preprocess_data(data_path)
    except FileNotFoundError:
        print(f"Error: Dataset {data_path} tidak ditemukan.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=5)
        model.fit(X_train, y_train)
        print("CI/CD Model training selesai.")

if __name__ == "__main__":
    main()
