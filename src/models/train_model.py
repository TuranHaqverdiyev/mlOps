import warnings
import pandas as pd
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from category_encoders import CatBoostEncoder
from xgboost import XGBClassifier

# Ignore warnings for cleaner output
warnings.filterwarnings("ignore")

# Custom Preprocessing Classes


class NumericTransformer(BaseEstimator, TransformerMixin):
    """Handles numeric columns: converts type, clips 'age', and imputes missing values."""

    def __init__(self, age_clip_min=18, age_clip_max=80):
        self.age_clip_min = age_clip_min
        self.age_clip_max = age_clip_max
        self.medians = {}

    def fit(self, X, y=None):
        X_copy = X.copy()
        for col in X_copy.columns:
            X_copy[col] = pd.to_numeric(X_copy[col], errors="coerce")
            if col == "age":
                X_copy[col] = X_copy[col].clip(self.age_clip_min, self.age_clip_max)
            self.medians[col] = X_copy[col].median()
        return self

    def transform(self, X, y=None):
        X_transformed = X.copy()
        for col in X_transformed.columns:
            X_transformed[col] = pd.to_numeric(X_transformed[col], errors="coerce")
            if col == "age":
                X_transformed[col] = X_transformed[col].clip(self.age_clip_min, self.age_clip_max)
            X_transformed[col] = X_transformed[col].fillna(self.medians[col])
        return X_transformed


class MonthlyDataTransformer(BaseEstimator, TransformerMixin):
    """Imputes missing values in monthly data with the median."""

    def __init__(self):
        self.medians = {}

    def fit(self, X, y=None):
        for col in X.columns:
            self.medians[col] = X[col].median()
        return self

    def transform(self, X, y=None):
        X_transformed = X.copy()
        for col in X_transformed.columns:
            X_transformed[col] = X_transformed[col].fillna(self.medians[col])
        return X_transformed


if __name__ == "__main__":
    print("Starting the model training process...")

    # 1. Load Data
    try:
        df = pd.read_parquet("multisim_dataset.parquet")
    except FileNotFoundError:
        print("Error: 'multisim_dataset.parquet' not found in the root directory.")
        exit()

    df = df.sample(frac=1, random_state=57).set_index("telephone_number")
    print("Data loaded and shuffled.")

    # 2. Split Data
    X = df.drop(columns="target")
    y = df["target"]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=57, stratify=y)

    # 3. Define Column Groups
    numeric_cols = ["age", "tenure", "age_dev", "dev_num"]
    categorical_cols = [
        "trf",
        "gndr",
        "dev_man",
        "device_os_name",
        "simcard_type",
        "region",
    ]
    monthly_cols = [col for col in X_train.columns if col.startswith("val")]

    # 4. Create Preprocessing Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", NumericTransformer(), numeric_cols),
            ("monthly", MonthlyDataTransformer(), monthly_cols),
            ("categorical", CatBoostEncoder(), categorical_cols),
        ],
        remainder="drop",
    )

    # 5. Define the Model with Best Parameters
    best_params = {
        "n_estimators": 334,
        "max_depth": 4,
        "learning_rate": 0.022287980174937348,
        "subsample": 0.6037684234853827,
        "colsample_bytree": 0.6099641378129602,
        "gamma": 1.0375078910112627,
        "lambda": 1.2483253701614754,
        "alpha": 2.270116860679173,
        "random_state": 57,
        "eval_metric": "logloss",
    }
    final_model = XGBClassifier(**best_params)

    # 6. Create and Train the Full Pipeline
    full_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", final_model)])

    print("Training the final model...")
    full_pipeline.fit(X_train, y_train)
    print("Model training complete.")

    # 7. Save the Model
    model_filename = "models/model_pipeline.joblib"
    joblib.dump(full_pipeline, model_filename)
    print(f"Model pipeline saved to '{model_filename}'")
