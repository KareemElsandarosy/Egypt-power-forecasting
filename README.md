# ⚡ Egypt Power Load Forecasting System
<div align="center">
  <h3>Intelligent hourly electricity demand prediction across all governorates</h3>
  <p>A Machine Learning system utilizing XGBoost, Random Forest, and Linear Regression to forecast electricity consumption in Egypt based on real-time weather data.</p>
</div>

---

## 📌 Project Overview
This project predicts the hourly electricity load (in Megawatts) for 27 Egyptian governorates. It features a professional, interactive web dashboard built with Streamlit that automatically fetches real weather data via Open-Meteo API and applies a trained Multi-Model Machine Learning pipeline to generate accurate forecasts up to 7 days ahead.

## 🚀 Features
- **Multi-Model Engine:** Compare live forecasts between Random Forest, XGBoost, and Linear Regression.
- **Dynamic Switching:** Instantly switch between models and see confidence margins (MAE & R2).
- **Real-time Weather:** 100% real hourly weather data (Temperature, Humidity, Wind Speed) fetched automatically.
- **Data Profiling:** In-depth interactive dataset dictionary, feature visualization, and descriptive statistics.
- **Professional UI:** Light-themed, fully responsive dashboard with Material Icons and interactive Plotly charts.

## 🛠️ Prerequisites
- Python 3.11+
- Git

## ⚙️ Installation & Setup (How to Run from Scratch)

Follow these exact steps to run the project on any new machine:

### 1. Clone the Repository
```bash
git clone https://github.com/KareemElsandarosy/Egypt-power-forecasting
cd egypt-power-forecasting
```

### 2. Install Dependencies
Open your terminal/command prompt and run:
```bash
pip install pandas numpy xgboost scikit-learn joblib streamlit plotly requests
```

### 3. Train the Models (Generate the Pipelines)
Before starting the web app, you must train the AI models. This script will train Linear Regression, Random Forest, and XGBoost on the dataset and save them into `model_pipeline.pkl`.
```bash
python scratch/train_models.py
```
*(Note: Wait until you see "✅ Pipeline saved successfully". This takes about 1-2 minutes.)*

### 4. Run the Web Application
Start the Streamlit dashboard:
```bash
streamlit run streamlit_app.py
```
Your browser will automatically open at `http://localhost:8501`.

---

## 📁 Repository Structure
- `streamlit_app.py`: The main frontend application and UI.
- `scratch/train_models.py`: The machine learning training script.
- `Egypt_Governorates_Load_Dataset_Advanced.csv`: The base dataset containing engineered features.
- `mapping_config.json`: Configurations mapping all 27 governorates to their base training regions.
- `model_pipeline.pkl`: (Generated after Step 3) Contains the serialized scikit-learn pipelines and model scores.

## ⚠️ Important Notes
- **Internet Connection Required:** The app must connect to the Open-Meteo API to fetch future weather data.
- **Synthetic Load Data:** While weather data is real, electricity load data is engineered (Synthetic) to simulate real consumption patterns using CAPMAS population weights and thermal correlation relationships for academic purposes.

---
**Made for Data Science Project - Term 2**
