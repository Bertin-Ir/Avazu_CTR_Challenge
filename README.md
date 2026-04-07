# Avazu_CTR_Challenge

##  Overview

This project addresses the problem of click-through rate (CTR) prediction using the Avazu CTR Prediction Challenge dataset. The goal is to estimate the probability that a user clicks on an advertisement given high-dimensional, sparse, and predominantly categorical data describing users, ads, devices, and context.

CTR prediction is a critical component of modern advertising systems, directly influencing ad targeting, ranking, and revenue optimization. Similar methodologies extend to recommendation systems, search ranking, and personalized content delivery.

---

##  Approach

We design a **progressive modeling pipeline**, moving from strong baselines to specialized architectures capable of capturing complex feature interactions.

- **Baseline Models:** Logistic Regression and XGBoost establish strong initial performance, enhanced through target encoding, handcrafted interactions, and polynomial features.
- **Feature Engineering:** Careful construction of interaction terms and temporal features, alongside aggressive handling of high-cardinality variables.
- **Dimensionality Control:** Implementation of **field-aware hashing and rare-category bucketing** to manage extreme feature cardinality while preserving interaction structure.
- **Advanced Models:** Transition to **Field-Aware Factorization Machines (FFM)** to systematically learn pairwise interactions conditioned on feature fields.
- **Deep Learning Extension:** Integration of a **Multi-Layer Perceptron (MLP)** to capture higher-order nonlinear patterns beyond pairwise interactions.
- **Ensembling:** Final model combines **FFM and MLP**, leveraging complementary strengths—structured interaction learning and deep nonlinear representation.

---

## Results

| Model                          | Log Loss on Validation data | 
|--------------------------------|--------:|
| Logistic Regression + Features | ~0.41   | 
| XGBoost + Engineered Features  | 0.402  | 
| **FFM + MLP Ensemble**         | **0.39** | 

---

## Tech Stack

- Python, NumPy, Pandas, Seaborn  
- Scikit-learn, XGBoost  
- PyTorch  
- Custom feature engineering & hashing pipelines  

---
