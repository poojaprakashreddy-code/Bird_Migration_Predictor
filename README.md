# 🦅 Bird Migration Success Predictor — Production ML Risk Engine

An end-to-end telemetry and environmental machine learning system built using modular design patterns to predict avian migration success rates using raw environmental telemetry.

## 🚀 Technical Core & Implementation
* **Pipeline Framework:** Built using scikit-learn pipeline objects encapsulating continuous scale transformers and categorical one-hot vectors to fully isolate training states and avoid data leakage.
* **Hyperparameter Optimization:** Utilized Stratified 5-fold cross-validated grid searches (`GridSearchCV`) targeting optimization of the Area Under Curve (AUC) metric space to handle implicit noise.
* **Robust Domain Interaction Modeling:** Engineered physical risk markers including a composite `Flight_Effort_Index` (factoring headwinds and physical travel scales against rest intervals) and an `Environmental_Stress_Index`.
* **Algorithmic Explainability:** Bypassed baseline tree-impurity biases by tracking accurate model evaluations via Permutation Feature Importance weights.

## 📊 Core Discovery Output
* The model evaluates dynamic environmental interactions, determining critical thresholds where wind speed and visibility constraints heavily impact migration success patterns regardless of species class or flock density.
* Integrating cross-validated hyperparameters safeguards training stability, maintaining robust classification performance boundaries without over-fitting the continuous feature set.

## 🛠️ Local Execution 
To run this pipeline locally inside your web browser:
1. Download `Bird_Migration_Predictor.ipynb` and `bird_migration_data.csv` to the same folder.
2. Launch your environment:
   ```bash
   jupyter notebook
