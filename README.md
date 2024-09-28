💳 Credit Card Fraud Detection Using Python 🚨

This project builds a machine learning model to detect fraudulent credit card transactions using a Logistic Regression and Random Forest classifier. The dataset is highly imbalanced, so we used SMOTE (Synthetic Minority Over-sampling Technique) to handle class imbalance and improve fraud detection.

🚀 Project Overview
The primary goal of this project is to build a classification model that can accurately classify transactions as either fraudulent or genuine.

Key steps:

Data Preprocessing: Normalizing the data and handling class imbalance with SMOTE.
Modeling: Training both Logistic Regression and Random Forest models to classify transactions.
Evaluation: Using performance metrics like precision, recall, F1-score, and the confusion matrix.
Visualization: Creating a confusion matrix and ROC curve for model evaluation.
📂 Dataset
The dataset used for this project is creditcard.csv, which contains:

Time: Time elapsed between the transaction and the first transaction in the dataset.
V1 - V28: Anonymized features derived from a PCA transformation.
Amount: Transaction amount.
Class: Target variable (0 = Non-Fraud, 1 = Fraud).
🛠️ Technologies Used
Python
pandas and NumPy for data manipulation.
Scikit-learn for machine learning algorithms.
imbalanced-learn for handling class imbalance (SMOTE).
Matplotlib and Seaborn for data visualization.

🧑‍💻 Installation and Setup
1. Clone the Repository
git clone https://github.com/Jeevanreddygopidi/CODSOFT-CREDIT-CARD-FRAUD-DETECTION.git
cd credit-card-fraud-detection
2. Create and Activate a Virtual Environment (recommended)
# Create virtual environment
python -m venv venv

# Activate the virtual environment
# For Windows:
venv\Scripts\activate

# For Mac/Linux:
source venv/bin/activate
3. Install Required Libraries
Install all necessary libraries using:

pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn
4. Run the Code
To run the code and see the model’s results:
python creditcard.py
⚙️ How It Works
Load the Dataset: We load the dataset using pandas and check for missing values and overall data structure.

Data Preprocessing:

Normalization: The feature columns are normalized using StandardScaler.
Class Imbalance: We use SMOTE to oversample the minority class (fraudulent transactions) and balance the dataset.
Model Training:

Logistic Regression: We train a Logistic Regression model and evaluate its performance.
Random Forest: We also train a Random Forest model, which generally performs better on imbalanced datasets.
Evaluation:

The models are evaluated using precision, recall, F1-score, and confusion matrices.
The ROC curve is plotted to assess how well the model distinguishes between the two classes.
📊 Results
Logistic Regression Performance:

Precision: X.XX
Recall: X.XX
F1-Score: X.XX
Random Forest Performance:

Precision: X.XX
Recall: X.XX
F1-Score: X.XX
The ROC curve showed that the Random Forest model performed better, with an AUC score of X.XX, indicating better fraud detection capabilities.

🔧 Next Steps
Experiment with other machine learning models like XGBoost or Gradient Boosting for even better performance.
Consider using cross-validation to better evaluate model performance across different training sets.

📞 Contact
If you have any questions, suggestions, or run into any issues, feel free to reach out:

Email: jeevanreddy.work@gmail.com

LinkedIn: http://www.linkedin.com/in/jeevan--reddy

You can also open an issue on this repository for further help or feedback.
📜 License
This project is licensed under the MIT License - see the LICENSE file for details.
