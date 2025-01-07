import pandas as pd
import pickle

# Tải mô hình
with open("./saved_models/svd_model.pkl", "rb") as f:
    model = pickle.load(f)

# Đọc dữ liệu
ratings_data = pd.read_csv("./data/ratings.csv")
products = pd.read_csv("./data/products.csv")

# Hàm gợi ý sản phẩm
def recommend_products(user_id, num_recommendations=5):
    all_product_ids = ratings_data['product_id'].unique()
    rated_products = ratings_data[ratings_data['user_id'] == user_id]['product_id']
    unrated_products = [prod for prod in all_product_ids if prod not in rated_products.values]
    predictions = [(prod, model.predict(user_id, prod).est) for prod in unrated_products]
    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions[:num_recommendations]

# Chạy gợi ý cho user_id = 1
user_id = 1
recommendations = recommend_products(user_id)

# Hiển thị kết quả
print(f"Gợi ý sản phẩm cho khách hàng {user_id}:")
for product_id, score in recommendations:
    product_info = products[products['product_id'] == product_id]
    print(f"Tên sản phẩm: {product_info['name'].values[0]}, Loại: {product_info['category'].values[0]}, Dự đoán điểm: {score:.2f}")
