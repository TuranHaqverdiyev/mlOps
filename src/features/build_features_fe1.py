import numpy as np
import pandas as pd
import joblib
import os
from sklearn.preprocessing import PowerTransformer, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import category_encoders as ce


def main():
    # 1. Define file paths
    input_filepath = "data/external/data_usage_production.parquet"
    processed_dir = "data/processed"
    models_dir = "models"

    # 2. Load and sample the dataset
    df_original = pd.read_parquet(input_filepath)
    df = df_original.sample(n=100000, random_state=57)
    print(f"Loaded and sampled 100,000 rows from {input_filepath}")

    # 3. Prepare data
    if "telephone_number" in df.columns:
        df.set_index("telephone_number", inplace=True)

    target = "data_compl_usg_local_m1"
    X = df.drop(target, axis=1)
    y = df[target]

    # 4. Define column groups and preprocessing pipelines
    numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("transformer", PowerTransformer(method="yeo-johnson")),
            ("scaler", RobustScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", ce.CatBoostEncoder(handle_unknown="value")),
        ]
    )

    # 5. Create the master preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, numerical_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )

    # 6. Fit and transform the data
    X_transformed = preprocessor.fit_transform(X, y)
    print("Preprocessor fitted on the entire dataset.")

    # 7. Save the fitted preprocessor object
    joblib.dump(preprocessor, os.path.join(models_dir, "preprocessor_fe1.joblib"))
    print(f"Preprocessor saved to '{os.path.join(models_dir, 'preprocessor_fe1.joblib')}'")

    # 8. Create and save the final processed DataFrame
    encoded_categorical_names = list(
        preprocessor.named_transformers_["cat"]["encoder"].get_feature_names_out()
    )
    feature_names = numerical_cols + encoded_categorical_names

    processed_df = pd.DataFrame(X_transformed, index=X.index, columns=feature_names).join(y)

    output_filepath = os.path.join(processed_dir, "data_usage_production.parquet")
    processed_df.to_parquet(output_filepath)
    print(f"Final processed dataset saved to '{output_filepath}'")


if __name__ == "__main__":
    main()
