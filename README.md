# Recommendation System

## Cấu trúc
- `data/`: Chứa dữ liệu sản phẩm và đánh giá.
- `src/`: Chứa mã nguồn chính.
- `saved_models/`: Lưu mô hình đã huấn luyện.

## Cài đặt
1. Tạo môi trường Python: `python -m venv venv`
2. Kích hoạt môi trường: 
   - Windows: `venv\Scripts\activate`
   - Linux/Mac: `source venv/bin/activate`
3. Cài đặt thư viện: `pip install -r requirements.txt`

## Cách chạy
1. Huấn luyện mô hình: `python src/train_model.py`
2. Chạy gợi ý: `python src/recommend.py`
