# 🧠 Customer Churn Prediction
This project focuses on predicting customer churn, a critical business metric that reflects when customers stop using a company's products or services. By leveraging machine learning techniques, this project aims to help businesses proactively identify customers at risk of leaving and take preventative action.

# 📂 Project Overview
Customer churn is a **key challenge** for subscription-based businesses. Retaining existing customers is often more cost-effective than acquiring new ones. Using a labeled churn dataset, this project builds a **predictive model** to classify whether a customer will churn or remain loyal.

# ✅ Goals:
Understand patterns that lead to customer churn.
* Build a machine learning pipeline for classification.
* Tune the model for optimal performance.
* Deploy the solution via a Dockerized app.

# 🧾 Dataset Description
The dataset used in this project includes customer-level information and a binary target label:
* Churned (1) – The customer has discontinued their subscription or service.
* Not Churned (0) – The customer is still active.

More info: https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset


# 🛠️ Tech Stack & Tools
* Language	Python
* ML Libraries	scikit-learn, pandas, numpy, feature-engine
* Optimization	Optuna 
* Evaluation	Cross-validation, F1-score
* Deployment	Docker, litserve
* Visualization	matplotlib, seaborn

# 📈 Model Development
The modeling process included the following steps:

### Data Preprocessing:
* Missing value handling
* Feature encoding
* Scaling

### Model Selection & Tuning:
* Cross-validation using StratifiedKFold
* Hyperparameter tuning with Bayesian Optimization for performance and efficiency

### Model Evaluation:
* Metrics used: Accuracy, Precision, Recall, F1-Score

# 🚢 Deployment
The model is deployed using a lightweight Dockerized Python app.

Docker Features:
* Python 3.12 slim base image
* Clean and optimized Dockerfile
* Port 4141 exposed for local interaction
* Mountable volume for models (/models)

To build and run the Docker container:
### Build the image
docker build -t churn-predictor .

### Run the container
docker run -p 4141:4141 -v $(pwd)/models:/models churn-predictor

### Agentic AI integration
The deployment allows an integration with Agentic AI workflows through an **MCP server**. It is available at ``hostname:4141/mcp`` aassuming that the service the above docker run command is executed.


# 🧪 Future Work
* Integrate API endpoints for real-time prediction
* Add logging and monitoring
* Develop a Streamlit or Gradio frontend
* Connect to real-time customer data pipelines