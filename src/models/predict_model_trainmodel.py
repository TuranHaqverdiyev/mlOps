import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


def main():

    # 1. Define file paths
    input_filepath = "data/processed/multisim_dataset.parquet"
    models_dir = "models"
    model_filepath = os.path.join(models_dir, "model_trainmodel.joblib")

    # 2. Load the processed dataset
    try:
        df = pd.read_parquet(input_filepath)
    except FileNotFoundError:
        print(f"Error: Processed data file not found at '{input_filepath}'.")
        print("Please run the build_features_trainmodel.py script first.")
        return

    # 3. Split the data to get the test set
    target = "target"
    X = df.drop(target, axis=1)
    y = df[target]

    # Use train_test_split to get the exact same test set as in training
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=57)

    # 4. Load the pre-trained model
    try:
        model = joblib.load(model_filepath)
        print(f"Model loaded successfully from '{model_filepath}'")
    except FileNotFoundError:
        print(f"Error: Trained model not found at '{model_filepath}'.")
        print("Please run the train_model_trainmodel.py script first.")
        return

    # 5. Make predictions on the test set
    y_pred = model.predict(X_test)

    # 6. Calculate evaluation metrics from the notebook
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # 7. Print the results
    print("\nModel Evaluation Results")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)


if __name__ == "__main__":
    main()
