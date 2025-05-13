# 🛡️ Insurance Claims Prediction Pipeline

## Overview
This project productionizes a machine learning model that predicts the likelihood of insurance claims. It supports:
- Daily batch predictions for new applications (~1200 per day)
- Monthly retraining on fresh data from the data warehouse
- A FastAPI interface to manually trigger and view predictions
- Full CI/CD pipeline, Dockerization, and Kubernetes CronJobs
- Deployment on Google Cloud Platform (GCP)

## Technologies Used
- Python
- FastAPI for REST API
- Pytest for unit testing
- Docker for containerization
- GitHub Actions for CI/CD
- Kubernetes CronJobs for job scheduling and deployment
- Google Cloud Platform (GCP) for cloud infrastructure

## Functionality

### Daily Predictions
- Automatically triggered each day using Kubernetes CronJob
- Loads daily data from the provided functions
- Applies preprocessing and runs the trained model
- Stores predictions and confidence scores in a structured format such as CSV (optionally a database in upcoming versions)

### Monthly Retraining
- Automatically triggered monthly
- Loads updated data from the provided functions
- Retrains model with same pipeline logic
- Saves updated model to local (optionally registered with MLflow in upcoming versions)

### FastAPI Interface
- Allows for single instance prediction testing using UI
- Supports visualization of prediction output of daily predictions

## Assumptions
- Daily application data is available in the warehouse in the following day 1 AM 
- Data includes a `timestamp` column to distinguish batches
- Model training logic does not need enhancement at this stage
- Same preprocessing pipeline is used consistently for inference and training (Assuming the data consistent over the time)
- Output format is structured for downstream use

## Business Considerations
- Supports underwriters in risk assessment by providing early claim likelihood
- Enables proactive customer handling during the cooling-off period
- Ensures retraining frequency balances performance and stability
- Predictions and retraining need to be automated, stable, and interpretable
- Data governance and quality are assumed to be handled upstream

## CI/CD and Automation

### GitHub Actions
- Triggers on code push and pull requests to main
- Runs unit tests and linting
- Builds and optionally pushes Docker image to registry
- Deploys to GCK via GitHub secrets and service account credentials

### Kubernetes CronJobs
- Daily prediction job scheduled in the following day 1 AM UTC
- Monthly retraining job scheduled on the first day of each month
- Secrets and configs managed using Kubernetes Secrets or environment variables

## Docker and Deployment
- Dockerfile creates a lightweight container for the full pipeline
- Containers deployed on GCP Kubernetes Engine
- Jobs triggered using K8s CronJobs
- Optionally integrates with Artifact Registry to store the docker image

## Answers to Key Questions

### What assumptions are you making?
- Data is available and clean before job runs
- Timestamp exists for temporal filtering
- Same data preprocessing and model logic is used for training and inference
- Prediction output can be stored in a flat file or sent downstream

### What are the business considerations?
- Speed of delivery during the 2-week cooling-off period
- Transparency of model predictions for underwriting
- Consistency of retraining for model performance
- Automation and minimal manual intervention

### Who would you talk to?
- Data Engineers for warehouse schema and availability SLAs
- Data Scientist to incorporate their work properly
- Project Manager to track the progress
- Claims or Product Owners or Underwriting for validation of prediction utility

### What is in scope vs. out of scope?

**In scope:**
- Refactoring notebook to modular pipeline
- Automating batch prediction and retraining
- Containerization, CI/CD, and deployment
- Creating an API interface

**Out of scope:**
- Improving data preprocessing steps or model performance
- Building upstream ETL pipelines


## Local Development Workflow

### Create virtual environment and install dependencies

```bash
python3 -m venv env  
source env/bin/activate  
pip install -r requirements.txt  
```

### Run FastAPI application for local API testing

```bash
uvicorn main:app --reload  
```

### Run prediction and retraining scripts independently

**Daily prediction:**

```bash
python jobs/daily_run.py  
```

**Monthly retraining:**

```bash
python jobs/monthly_run.py  
```

### Execute unit tests for core modules and utilities

```bash
pytest tests/  
```

## Further Improvements

- Replace Kubernetes CronJobs with **Apache Airflow DAGs** or **Cloud Functions** for more robust, scalable, and observable orchestration.
- Use **MLflow** to track and register models instead of saving to local storage for better model versioning and transparency.
- Add **model monitoring and data drift detection** to ensure model reliability over time and trigger alerts or retraining when necessary.
- Save prediction results directly into a **cloud database** or table (e.g., BigQuery, CloudSQL) instead of flat CSV files for better integration with downstream processes.
