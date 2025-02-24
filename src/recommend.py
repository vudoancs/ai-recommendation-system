import pandas as pd
import pickle
import implicit
import numpy as np
from scipy.sparse import csr_matrix
from collections import Counter


def get_user_preferences(user_id, ratings_df, products_df):
    """Analyze user's category preferences based on their ratings"""
    user_ratings = ratings_df[ratings_df["user_id"] == user_id]
    user_products = user_ratings.merge(products_df, on="product_id")

    # Calculate average rating per category
    category_ratings = (
        user_products.groupby("category")["rating"].agg(["mean", "count"]).reset_index()
    )
    return category_ratings


def get_recommendations(user_id, model, ratings_df, products_df, n=10):
    # Convert IDs to internal indices
    user_ids = ratings_df["user_id"].astype("category")
    product_ids = ratings_df["product_id"].astype("category")
    ratings = ratings_df["rating"].astype(float)

    # Create sparse matrix
    sparse_ratings = csr_matrix((ratings, (user_ids.cat.codes, product_ids.cat.codes)))

    # Get user index
    user_idx = user_ids.cat.categories.get_loc(user_id)

    # Get more recommendations than needed to allow for filtering
    ids, scores = model.recommend(
        user_idx, sparse_ratings[user_idx], N=n * 2, filter_already_liked_items=True
    )

    # Convert back to product IDs
    recommended_products = [product_ids.cat.categories[idx] for idx in ids]

    # Get user preferences
    user_preferences = get_user_preferences(user_id, ratings_df, products_df)

    # Filter and sort recommendations based on user preferences
    recommendations = []
    for prod_id, score in zip(recommended_products, scores):
        product_info = products_df[products_df["product_id"] == prod_id].iloc[0]
        category_pref = (
            user_preferences[user_preferences["category"] == product_info["category"]][
                "mean"
            ].iloc[0]
            if len(
                user_preferences[
                    user_preferences["category"] == product_info["category"]
                ]
            )
            > 0
            else 0
        )

        # Adjust score based on category preference
        adjusted_score = score * (1 + category_pref / 5)
        recommendations.append(
            {
                "product_id": prod_id,
                "name": product_info["name"],
                "category": product_info["category"],
                "score": adjusted_score,
            }
        )

    # Sort by adjusted score and take top n
    recommendations.sort(key=lambda x: x["score"], reverse=True)
    recommendations = recommendations[:n]

    # Normalize scores to percentages
    scores = np.array([r["score"] for r in recommendations])
    normalized_scores = (scores - scores.min()) / (scores.max() - scores.min()) * 100

    for rec, norm_score in zip(recommendations, normalized_scores):
        rec["confidence"] = norm_score

    return recommendations


if __name__ == "__main__":
    # Load model and data
    with open("./saved_models/implicit_model.pkl", "rb") as f:
        model = pickle.load(f)

    ratings_df = pd.read_csv("data/ratings.csv")
    products_df = pd.read_csv("data/products.csv")

    # Example: Get recommendations for user 1
    recommendations = get_recommendations(1, model, ratings_df, products_df)
    print("\nRecommended products for user 1:")
    for rec in recommendations:
        print(f"Product: {rec['name']}")
        print(f"Category: {rec['category']}")
        print(f"Confidence: {rec['confidence']:.1f}%")
        print("---")
