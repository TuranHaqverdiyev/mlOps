import pandas as pd
import os
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from category_encoders import CatBoostEncoder
from sentence_transformers import SentenceTransformer


class RamenVarietyTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, pca_components=50, n_clusters=5, random_state=57):
        self.pca_components = pca_components
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.pca = PCA(n_components=self.pca_components, random_state=self.random_state)
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)

    def fit(self, X, y=None):
        # The fit method learns the PCA and KMeans models from the text data
        variety_texts = X["Variety"].tolist()
        embeddings = self.embedding_model.encode(variety_texts, show_progress_bar=False)
        embeddings_reduced = self.pca.fit_transform(embeddings)
        self.kmeans.fit(embeddings_reduced)
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        variety_texts = X_copy["Variety"].tolist()

        # Generate embeddings
        embeddings = self.embedding_model.encode(variety_texts, show_progress_bar=False)

        # Create embedding DataFrame
        embedding_df = pd.DataFrame(embeddings, index=X_copy.index)
        embedding_df.columns = [f"variety_emb_{i}" for i in range(embeddings.shape[1])]

        # Reduce dimensionality and create cluster feature
        embeddings_reduced = self.pca.transform(embeddings)
        X_copy["variety_cluster"] = self.kmeans.predict(embeddings_reduced)

        # Combine with original data (minus the text column) and the full embeddings
        return pd.concat([X_copy.drop(columns=["Variety"]), embedding_df], axis=1)


def main():

    # 1. Define file paths
    input_filepath = "data/external/ramen-ratings.csv"
    processed_dir = "data/processed"
    models_dir = "models"

    # 2. Load and perform initial cleaning
    df = pd.read_csv(input_filepath)
    df.set_index("Review #", inplace=True)

    df["Stars"] = pd.to_numeric(df["Stars"], errors="coerce")
    df.dropna(subset=["Stars"], inplace=True)

    df["Top Ten"].fillna("0", inplace=True)
    df["is_top_ten"] = df["Top Ten"].apply(lambda x: 0 if x == "0" else 1)
    df.drop("Top Ten", axis=1, inplace=True)
    print("Initial data cleaning complete.")

    # 3. Separate features and target
    target = "Stars"
    X = df.drop(target, axis=1)
    y = df[target]

    # 4. Define the feature engineering pipeline
    categorical_features = ["Brand", "Style", "Country"]

    # The preprocessor applies transformers to different columns
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical_encoder", CatBoostEncoder(), categorical_features),
            ("variety_transformer", RamenVarietyTransformer(), ["Variety", "is_top_ten"]),
        ],
        remainder="passthrough",
    )

    # 5. Fit the preprocessor and transform the data
    X_transformed = preprocessor.fit_transform(X, y)
    print("Feature engineering pipeline fitted and data transformed.")

    # 6. Save the fitted preprocessor object
    joblib.dump(preprocessor, os.path.join(models_dir, "preprocessor_fe2.joblib"))
    print(f"Preprocessor saved to '{os.path.join(models_dir, 'preprocessor_fe2.joblib')}'")

    # 7. Create and save the final processed DataFrame
    cat_encoded_names = categorical_features
    # From RamenVarietyTransformer, we get 'is_top_ten', 'variety_cluster', and 384 embedding columns
    variety_emb_names = [f"variety_emb_{i}" for i in range(384)]
    other_names = ["is_top_ten", "variety_cluster"]

    all_feature_names = cat_encoded_names + other_names + variety_emb_names

    processed_df = pd.DataFrame(X_transformed, index=X.index, columns=all_feature_names).join(y)

    output_filepath = os.path.join(processed_dir, "ramen-ratings.parquet")
    processed_df.to_parquet(output_filepath)
    print(f"Final processed dataset saved to '{output_filepath}'")


if __name__ == "__main__":
    main()
