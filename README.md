# Predicting-Loan-Repayment-Risk-Using-Machine-Learning
Business Scenario
Financial institutions often reject credit applicants due to limited or non-existent credit history. This project aims to address that by building a predictive model using alternative data such as telco and transactional info, helping lenders assess the likelihood of loan repayment more fairly and effectively.
Our work is inspired by the Home Credit Default Risk Kaggle competition, where the challenge is to predict whether a client will repay a loan or default, based on a massive multi-source dataset.


🎯 Project Objective
To develop a robust machine learning model that can predict credit default risk by:
•	Handling millions of rows and high-dimensional data
•	Managing missing values and imbalanced classes
•	Performing feature engineering across multiple datasets
•	Training, tuning, and evaluating models using cross-validation
•	Maximizing predictive performance (AUC)


👥 User Stories
•	Loan Officer: I want a system that accurately predicts credit default risk to improve loan approval decisions.
•	Applicant: I want fairer credit evaluations, even with limited credit history.
•	Data Scientist: I want to use complex, real-world data to solve high-impact financial challenges.


🧠 Methodology Breakdown
📘 Phase 1: Baseline Model Building
•	Explored key tables: application_train, bureau, credit_card_balance, previous_application, etc.
•	Performed data cleaning, missing value handling, and basic preprocessing.
•	Built baseline models using Logistic Regression, with minimal feature engineering.
•	Applied Label Encoding and One-Hot Encoding for categorical variables.
Outcome: Built a working baseline pipeline and achieved initial accuracy scores. Identified areas for improvement like feature extraction and aggregation.


🛠️ Phase 2: Feature Engineering & Data Integration
•	Joined datasets using SK_ID_CURR and SK_ID_PREV keys.
•	Summarized transactional tables using group-by aggregates (e.g., max, mean, min).
•	Created derived features like:
–	Payment delay = DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT
–	Debt ratio = AMT_CREDIT_SUM_DEBT / AMT_CREDIT_SUM
•	Dealt with imbalanced classes using SMOTE and resampling techniques.
•	Evaluated different models: XGBoost, LightGBM, RandomForest.
Outcome: Enhanced feature richness and significantly boosted model AUC and F1 scores.


🧪 Phase 3: Final Modeling and Submission
•	Refined final models with:
–	Cross-validation using StratifiedKFold
–	Grid Search for hyperparameter tuning
•	Built a custom pipeline combining:
–	Feature selection (top 50 features)
–	Scaling, encoding, and modeling
•	Final model: LightGBM, tuned for best AUC
•	Submitted predictions to Kaggle with consistent results
Outcome: Successfully integrated all phases into a reproducible pipeline with competitive performance on the leaderboard.


⚙️ Technology Stack
Layer	Tools & Libraries
Data Processing	Pandas, NumPy, Scikit-learn
Modeling	LightGBM, XGBoost, RandomForest
Visualization	Seaborn, Matplotlib
Validation	KFold CV, GridSearchCV
Feature Engg.	Aggregates, Ratios, Temporal features


📌 Key Takeaways
•	Efficient handling of large, sparse, and relational datasets
•	Built and optimized a high-performing ML pipeline
•	Improved model interpretability using engineered features
•	Applied real-world credit risk modeling using data science best practices


📈 Final Impact
•	Learned end-to-end data science workflow
•	Improved AUC from baseline to competitive model
•	Gained practical experience in tackling real financial datasets

