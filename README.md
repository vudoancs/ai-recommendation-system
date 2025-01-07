# Recommendation System

## Structure
- `data/`: Contains product and rating data.
- `src/`: Contains the main source code.
- `saved_models/`: Stores the trained models.

## Setup
1. Create a Python environment: `python -m venv venv`
2. Activate the environment:
   - Windows: `venv\Scripts\activate`
   - Linux/Mac: `source venv/bin/activate`
3. Install libraries: `pip install -r requirements.txt`

## How to Run
1. Train the model: `python src/train_model.py`
2. Get recommendations: `python src/recommend.py`
