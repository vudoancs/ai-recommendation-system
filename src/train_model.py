import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
import pickle
import os

print("Current working directory:", os.getcwd())

# Đọc dữ liệu
ratings_data = pd.read_csv("./data/ratings.csv")
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings_data[['user_id', 'product_id', 'rating']], reader)

# Huấn luyện mô hình
trainset, testset = train_test_split(data, test_size=0.2)
model = SVD()
model.fit(trainset)

# Lưu mô hình
with open("./saved_models/svd_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("Đã lưu mô hình vào saved_models/svd_model.pkl")
