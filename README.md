# 🍷 MLflow End-to-End Machine Learning Pipeline

This project demonstrates an **end-to-end ML lifecycle** using **MLflow** with multiple machine learning models, remote tracking server integration, and automated model registration.

The system trains, evaluates, and tracks several regression and classification models on the Wine Quality dataset.

---

# 🚀 Features

✅ Multiple Regression Models Training
✅ Classification Pipeline
✅ PCA + Machine Learning Pipeline
✅ MLflow Experiment Tracking
✅ Remote AWS MLflow Server Integration
✅ Model Registry Automation
✅ Metrics Logging & Comparison

---

# 📊 Models Implemented

## Regression Models

* Linear Regression
* ElasticNet Regression
* Random Forest Regressor
* Decision Tree Regressor

## Classification Model

* Logistic Regression (Binary Wine Quality Classification)

## Dimensionality Reduction

* PCA + Linear Regression

---

# 📂 Dataset

Wine Quality Dataset from MLflow repository:

```
https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv
```

Target Variable:

```
quality
```

---

# ⚙️ Tech Stack

* Python
* Scikit-Learn
* MLflow
* Pandas
* NumPy
* AWS EC2 (Remote Tracking Server)

---

# 🧠 MLflow Capabilities Used

* Experiment Tracking
* Parameter Logging
* Metrics Logging
* Model Logging
* Model Registry
* Remote Tracking Server

---

# 📁 Project Structure

```
MLflow-Wine-Project/
│── example.py
│── requirements.txt
│── README.md
```

---

# 🔥 Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/mlflow-wine-project.git
cd mlflow-wine-project
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Run Project

```bash
python example.py
```

---

# 🌐 MLflow Tracking Server

The project uses a remote MLflow server:

```
http://ec2-54-147-36-34.compute-1.amazonaws.com:5000/
```

You can visualize experiments using:

```bash
mlflow ui
```

---

# 📈 Metrics Logged

## Regression Metrics

* RMSE
* MAE
* R² Score

## Classification Metrics

* Accuracy

---

# ☁️ Model Registry

Automatically registered models:

```
LinearRegressionWineModel
ElasticNetWineModel
RandomForestWineModel
DecisionTreeWineModel
LogisticWineModel
PCAWineModel
```

---

# 🏆 Learning Outcomes

This project demonstrates:

* End-to-End ML Pipeline
* Experiment Tracking at Scale
* Model Comparison & Selection
* Production-ready ML Workflow

---

# 👨‍💻 Author

Ashwani Yadav

---

# ⭐ Support

If you found this useful, please give this repository a ⭐.
