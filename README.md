# Water Quality Prediction

This project was developed as part of a **AICTE Machine Learning internship**.It uses specifically `MultiOutputRegressor` wrapped around a `RandomForestRegressor` for multiple water quality parameters.It predict water pollution levels based on the year and station ID, and evaluates whether the water is safe for consumption.

---

## 🚀 Live Demo

Check out 👉 (https://water-quality-pol-predictor.streamlit.app/)

---

## Overview
In this project:

- Collected and preprocessed real-world water quality datasets
- Used supervised machine learning for multi-target regression
- Built a pipeline using `MultiOutputRegressor` with `RandomForestRegressor`
- Evaluated the model using appropriate regression metrics

---

## 🔍 Features

- Predicts levels of key water pollutants:
  - BSK5, O2, NO3, NO2, SO4, PO4, CL
- Assesses drinking water safety based on environmental thresholds
- Clean, interactive web interface using **Streamlit**
- Model trained with **scikit-learn**, saved via **joblib**.

---

## Model Performance

The model was evaluated using:

- **Mean Squared Error (MSE)**
- **R² Score**
  
---

> **Model file:** [`pollution_model.pkl`](https://github.com/Sumeet010/Water-Quality-Prediction/blob/main/pollution_model.pkl)

---

## 🛠 Tech Stack

- **Python 3.10+**
- **Jupyter Notebook** (Model training)
- **Streamlit** (Model Deployment)
- **scikit-learn** (ML Model)
- **Pandas, NumPy**
- **Git LFS** (Model storage)

---
