# 🚀 AeroGuard - Predictive Maintenance System

**AeroGuard** is a machine learning-based predictive maintenance system designed to estimate the **Remaining Useful Life (RUL)** of aircraft engines using NASA’s CMAPSS dataset. The system helps anticipate engine failures in advance, enabling proactive maintenance and enhancing safety and operational efficiency.

---

## 📘 Project Overview

AeroGuard provides a complete data-driven pipeline to process engine sensor data, identify key indicators of degradation, and forecast how many operational cycles remain before maintenance is required. It offers both a backend predictive model and a user-friendly interface built with **Streamlit**.

---

## 🧠 Techniques & Methodology

### 🔹 1. Data Preprocessing
- Reads and processes the CMAPSS FD001 dataset (`train`, `test`, and `RUL` files).
- Computes Remaining Useful Life (RUL) for training data using each engine's max cycle.
- Aligns the last cycle of each test engine with its true RUL.

### 🔹 2. Feature Selection
- Trains a **Random Forest Regressor** to assess the importance of sensor and operational features.
- Automatically selects the **top 10 most relevant features** based on learned importance scores.

### 🔹 3. Model Architecture
- Uses a **hybrid ensemble** approach combining:
  - **Random Forest Regressor** (optimized with bootstrap, limited depth, and square-root feature sampling).
  - **XGBoost Regressor** (with controlled depth and learning rate).
- Final prediction is the **average of both model outputs**.

### 🔹 4. Prediction Correction
- Applies a fixed **correction factor (23.66)** to adjust for historical model overestimation.
- Ensures no negative RUL predictions by clipping at zero.

---

## 📊 Model Evaluation

The system calculates and displays the following performance metrics:

- **R² Score** – Measures prediction fit to actual RUL values.
- **Root Mean Squared Error (RMSE)** – Penalizes larger errors.
- **Mean Absolute Error (MAE)** – Average magnitude of prediction errors.

A line plot visualizes **Actual vs Predicted RUL**, and engine-wise results are shown in a tabular format.

---

## 💡 Key Highlights

- ✅ Supports real-time file upload and prediction via **Streamlit UI**.
- ✅ Intelligent feature selection using model-based importance ranking.
- ✅ Ensemble modeling for robustness and improved accuracy.
- ✅ Visualization of results and performance for interpretability.
- ✅ Modular design for further tuning or integration.

---

## 📂 Dataset

- **Used Subset**: FD001 from the CMAPSS dataset.
- **Source**: NASA Prognostics Center of Excellence  
  [Link to dataset](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository)

---

## 📓 Additional Resources

- `Pdm_script.ipynb`: Development notebook with exploratory experiments, model testing, and intermediate results.

---

## 📬 Acknowledgment

Developed with the goal of enhancing fault prediction systems in aviation through interpretable and reliable machine learning techniques.

---

