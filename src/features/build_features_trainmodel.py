import pandas as pd
import joblib
import os
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from category_encoders import CatBoostEncoder


class NumericTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, age_clip_min=18, age_clip_max=80):
        self.age_clip_min = age_clip_min
        self.age_clip_max = age_clip_max
        self.medians_ = {}

    def fit(self, X, y=None):
        X_copy = X.copy()
        for col in X_copy.columns:
            X_copy[col] = pd.to_numeric(X_copy[col], errors="coerce")
            if col == "age":
                X_copy[col] = X_copy[col].clip(self.age_clip_min, self.age_clip_max)
            self.medians_[col] = X_copy[col].median()
        return self

    def transform(self, X, y=None):
        X_transformed = X.copy()
        for col in X_transformed.columns:
            X_transformed[col] = pd.to_numeric(X_transformed[col], errors="coerce")
            if col == "age":
                X_transformed[col] = X_transformed[col].clip(self.age_clip_min, self.age_clip_max)
            X_transformed[col] = X_transformed[col].fillna(self.medians_[col])
        return X_transformed

    def get_feature_names_out(self, input_features=None):
        # This method is required for compatibility with ColumnTransformer
        return input_features


class MonthlyDataTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.medians_ = {}

    def fit(self, X, y=None):
        for col in X.columns:
            self.medians_[col] = X[col].median()
        return self

    def transform(self, X, y=None):
        X_transformed = X.copy()
        for col in X_transformed.columns:
            X_transformed[col] = X_transformed[col].fillna(self.medians_[col])
        return X_transformed

    def get_feature_names_out(self, input_features=None):
        # This method is required for compatibility with ColumnTransformer
        return input_features


def main():

    # 1. Define file paths
    input_filepath = "data/external/multisim_dataset.parquet"
    processed_dir = "data/processed"
    models_dir = "models"

    # 2. Load and sample the dataset
    df = pd.read_parquet(input_filepath)
    df = df.sample(frac=1, random_state=57).set_index("telephone_number")
    print(f"Loaded and sampled data from {input_filepath}")

    # 3. Prepare data for preprocessing
    target = "target"
    X = df.drop(target, axis=1)
    y = df[target]

    # 4. Define column groups and the complete preprocessor
    numeric_cols = ["age", "tenure", "age_dev", "dev_num"]
    categorical_cols = ["trf", "gndr", "dev_man", "device_os_name", "simcard_type", "region"]
    monthly_cols = [col for col in X.columns if col.startswith("val")]

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", NumericTransformer(), numeric_cols),
            ("monthly", MonthlyDataTransformer(), monthly_cols),
            ("categorical", CatBoostEncoder(), categorical_cols),
        ],
        remainder="drop",
    )

    # 5. Fit the preprocessor on the data and transform it
    X_transformed = preprocessor.fit_transform(X, y)
    print("Preprocessor fitted on the dataset.")

    # 6. Save the fitted preprocessor object
    joblib.dump(preprocessor, os.path.join(models_dir, "preprocessor_trainmodel.joblib"))
    print(f"Preprocessor saved to '{os.path.join(models_dir, 'preprocessor_trainmodel.joblib')}'")

    # 7. Create and save the final processed DataFrame
    feature_names = preprocessor.get_feature_names_out()
    processed_df = pd.DataFrame(X_transformed, index=X.index, columns=feature_names).join(y)

    output_filepath = os.path.join(processed_dir, "multisim_dataset.parquet")
    processed_df.to_parquet(output_filepath)
    print(f"Final processed dataset saved to '{output_filepath}'")


if __name__ == "__main__":
    main()
