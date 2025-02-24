import pandas as pd
import implicit
from scipy.sparse import csr_matrix
import pickle
import os
import numpy as np

print("Current working directory:", os.getcwd())


def prepare_data(ratings_df, products_df):
    # Add weight to ratings based on product category popularity
    category_popularity = ratings_df.merge(products_df, on="product_id")[
        "category"
    ].value_counts()
    category_weights = 1 + np.log1p(category_popularity) / np.log1p(
        category_popularity.max()
    )

    # Create category weight mapping
    category_weight_map = category_weights.to_dict()

    # Apply category weights to ratings
    weighted_ratings = ratings_df.merge(products_df, on="product_id").apply(
        lambda x: x["rating"] * category_weight_map[x["category"]], axis=1
    )

    return weighted_ratings


def train_model(ratings_df, products_df):
    # Prepare weighted ratings
    weighted_ratings = prepare_data(ratings_df, products_df)

    # Convert ratings to sparse matrix format
    users = ratings_df["user_id"].astype("category").cat.codes
    items = ratings_df["product_id"].astype("category").cat.codes

    # Create sparse matrix
    sparse_ratings = csr_matrix((weighted_ratings, (users, items)))

    # Initialize and train model with improved parameters
    model = implicit.als.AlternatingLeastSquares(
        factors=150,  # increased number of latent factors
        regularization=0.01,  # reduced regularization
        alpha=40,  # confidence scaling factor
        iterations=50,  # increased iterations
        calculate_training_loss=True,
        random_state=42,  # for reproducibility
    )
    model.fit(sparse_ratings)

    return model


if __name__ == "__main__":
    # Load data
    ratings_df = pd.read_csv("data/ratings.csv")
    products_df = pd.read_csv("data/products.csv")

    # Train model
    model = train_model(ratings_df, products_df)

    # Create directory if it doesn't exist
    os.makedirs("./saved_models", exist_ok=True)

    # Save model
    with open("./saved_models/implicit_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Model saved to saved_models/implicit_model.pkl")
