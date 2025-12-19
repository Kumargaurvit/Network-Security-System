# 🛡️ Network Security System – Phishing Website Detection

## 📌 Project Overview

The **Network Security System** is a machine learning–based web application designed to **detect whether a website is legitimate or a phishing site** based on extracted URL and network-related features.

The system automates the **end-to-end ML pipeline**, including:
- Data ingestion from MongoDB
- Data validation and drift detection
- Data transformation
- Model training and evaluation
- Prediction through a FastAPI web interface

It is built with **production-grade architecture**, following modular design, logging, exception handling, and CI/CD readiness.

---

## 🎯 Problem Statement

Phishing websites pose a serious cybersecurity threat by mimicking legitimate websites to steal sensitive information.  
This project aims to **classify website data as either legitimate or phishing**, helping organizations and users improve network security.

---

## 🧠 Solution Approach

1. **Dataset**
   - Phishing website dataset stored in MongoDB
   - Features extracted from URLs and website metadata

2. **Machine Learning Pipeline**
   - Data Ingestion
   - Data Validation (Schema & Drift Detection)
   - Data Transformation (Preprocessing & Imputation)
   - Model Training (Classification models)
   - Model Evaluation
   - Prediction & Output generation

3. **Web API**
   - FastAPI-based REST service
   - Upload URL and receive phishing predictions

---

## ⚙️ Tech Stack Used

### 🔹 Programming & Frameworks
- Python 3.13
- FastAPI
- Uvicorn

### 🔹 Machine Learning
- Scikit-learn
- NumPy
- Pandas
- Pickle

### 🔹 Database
- MongoDB Atlas
- PyMongo

### 🔹 DevOps & MLOps
- GitHub Actions
- Docker-ready architecture
- MLFlow and DAGsHub
- Data Drift Detection
- Logging & Exception Handling

### 🔹 Cloud Deployment
- Amazon S3 Bucket (Artifact and Model Storage)
- Amazon ECR (Docker Image Repository)
- Amazon EC2 Instance (Web App Deployment)

---