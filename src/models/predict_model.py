import pandas as pd
import joblib

if __name__ == "__main__":
    print("Loading model and new data for prediction")

    # Define file paths
    MODEL_PATH = "models/model_pipeline.joblib"
    NEW_DATA_PATH = "data/processed/new_customers.parquet"

    # 1. Load the saved model pipeline
    try:
        model_pipeline = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        print(f"Error: Model not found at '{MODEL_PATH}'.")
        print("Please run train_model.py first.")
        exit()

    # 2. Load the new, unseen data from the parquet file
    try:
        new_data = pd.read_parquet(NEW_DATA_PATH)
    except FileNotFoundError:
        print(f"Error: New data file not found at '{NEW_DATA_PATH}'.")
        print("You may need to create a sample prediction file first.")
        exit()

    # 3. Make predictions
    # The telephone_number is the index, so we pass the rest of the data
    predictions = model_pipeline.predict(new_data)

    # 4. Display the results
    new_data["prediction"] = predictions

    print("\nPrediction Results")
    print("New Customer Data(for first 5):")
    print(new_data.iloc[:, :5].join(new_data["prediction"]))
