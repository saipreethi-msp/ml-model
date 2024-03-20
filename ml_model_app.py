import streamlit as st
import pandas as pd
from ydata_profiling import ProfileReport
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore") 

from streamlit_pandas_profiling import st_profile_report

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,r2_score,mean_absolute_percentage_error,precision_score,recall_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

home = '''
        ## Welcome to the Machine Learning Model App!

        This web application allows you to perform various tasks related to machine learning models, 
        including data profiling, model selection, and evaluation.

        ### About the App:

        The app provides a user-friendly interface to upload datasets, perform data profiling to understand 
        the characteristics of your data, select machine learning models based on the target variable type, 
        train the selected models, and evaluate their performance.

        ### Models Available:

        - **Logistic Regression**: A linear model used for binary classification tasks.
        - **Linear Regression**: A linear model used for regression tasks.
        - **Decision Tree**: A tree-based model used for both classification and regression tasks.
        - **Random Forest**: An ensemble of decision trees used for both classification and regression tasks.
        - **XGBoost**: An optimized gradient boosting library used for both classification and regression tasks.

        ### Usage:

        - Click on **Data Profiling** to upload a dataset and generate a data profiling report.
        - Click on **Model Selection** to choose a model, train it using your dataset, and evaluate its performance.

        Start exploring by selecting an option from the sidebar menu!
        '''
# Load dataset function
def load_dataset():
    uploaded_file = st.file_uploader("Upload a dataset", type=["csv", "xlsx"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)  # Change read_csv to read_excel for xlsx files
            return df
        except Exception as e:
            st.error(f"Error: {e}")
    return None

# Data profiling page
def data_profiling():
    st.title("Data Profiling")
    df = load_dataset()
    if df is not None:
        st.write("### Data Exploration")
        st.write("#### Sample of Data:")
        st.write(df.head())
        
        st.write("#### Data Profiling Report:")
        profile = ProfileReport(df, explorative=True)
        st_profile_report(profile)
        st.balloons()

# Function to select target variable
def select_target_variable(df):
    st.write("### Select Target Variable")
    target_variable = st.selectbox("Select the target variable:", df.columns)
    return target_variable

# Model selection page
def model_selection(df):
    st.title("Model Selection")
    if df is not None:
        target_variable = select_target_variable(df)
        st.write(f"Selected target variable: {target_variable}")

        # Allow the user to select the type of the target variable
        target_type = st.radio("Select the type of target variable:", ("Categorical", "Continuous"))
        st.write(f"Target variable type: {target_type}")

        # Display models based on target variable type
        if target_type == 'Categorical':
            models = ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost"]
        else:
            models = ["Linear Regression", "Decision Tree", "Random Forest", "XGBoost"]

        selected_model = st.selectbox("Select model to train:", models)

        if st.button("Train Model"):
            model_training(df, target_variable, target_type, selected_model)

# Model training function
def model_training(df, target_variable, target_type, selected_model):
    st.write(f"Selected model: {selected_model}")

    # Split data into features and target variable
    X = df.drop(columns=[target_variable])
    y = df[target_variable]

    # Preprocess data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_pd, X_test_pd = preprocess_data(X_train, X_test)

    # Train selected model
    if selected_model == "Logistic Regression":
        task_type = 'classification'
        model = train_logistic_regression(X_train_pd, y_train, task=task_type)
    elif selected_model == "Linear Regression":
        task_type = 'regression'
        model = train_linear_regression(X_train_pd, y_train)
    elif selected_model == "Decision Tree":
        task_type = 'classification' if target_type == 'Categorical' else 'regression'
        model = train_decision_tree(X_train_pd, y_train, task_type)
    elif selected_model == "Random Forest":
        task_type = 'classification' if target_type == 'Categorical' else 'regression'
        model = train_random_forest(X_train_pd, y_train, task_type)
    elif selected_model == "XGBoost":
        task_type = 'classification' if target_type == 'Categorical' else 'regression'
        model = train_xg_boost(X_train_pd, y_train, task_type)

    st.write(f"Model trained successfully.")
    
    # Checkbox to show results

    st.title("Results")
    evaluate_model(model, X_test_pd, y_test, task_type)

    
# Evaluation function
def evaluate_model(model, X_test, y_test, task_type):
    st.write("Model Evaluation")
    if isinstance(model, RandomizedSearchCV):
        st.write("Best Hyperparameters:")
        st.write(model.best_params_)
        model = model.best_estimator_
    y_pred = model.predict(X_test)
    unique_labels = np.unique(y_test)
    if task_type == 'classification':
        if len(unique_labels) == 2:
            pos_label = unique_labels[0]
        else:
            pos_label = None
        # Calculate evaluation metrics for classification
        accuracy = round(accuracy_score(y_test, y_pred), 2)
        precision = round(precision_score(y_test, y_pred, average='micro'), 2)  # Using micro average for multiclass
        recall = round(recall_score(y_test, y_pred, average='micro'), 2)  # Using micro average for multiclass
        confusion_mat = confusion_matrix(y_test, y_pred)
        st.write("Confusion Matrix:")
        # Create a heatmap plot for the confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title('Confusion Matrix')
        st.pyplot(plt)  # Pass plt directly to st.pyplot()
        st.write("Accuracy:", accuracy)
        st.write("Precision:", precision)
        st.write("Recall:", recall)
        
        # Display classification report
        st.write("Classification Report:")
        report = classification_report(y_test, y_pred, target_names=unique_labels, output_dict=True)
        for key in report:
            if isinstance(report[key], dict):
                for k, v in report[key].items():
                    report[key][k] = round(v, 2)
        st.write(pd.DataFrame(report).transpose())

    else:
        # Calculate evaluation metrics for regression
        r2 = round(r2_score(y_test, y_pred), 2)
        mse = round(mean_squared_error(y_test, y_pred), 2)
        mape = round(mean_absolute_percentage_error(y_test, y_pred), 2)
        st.write("R-squared:", r2)
        st.write("Mean Squared Error (MSE):", mse)
        st.write("Mean Absolute Percentage Error (MAE):", mape)


def preprocess_data(X_train, X_test):
    # Identify numeric and categorical columns
    numeric = X_train.select_dtypes(include=['int64', 'float64']).columns
    categoric = X_train.select_dtypes(include=['object']).columns

    # Create transformers for numeric
    numeric_transform = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Create transformers for character
    character_transform = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Create a preprocesser
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transform, numeric),
            ('cat', character_transform, categoric)
        ]
    )
    X_train_pd = preprocessor.fit_transform(X_train)
    X_test_pd = preprocessor.transform(X_test)
    
    return X_train_pd, X_test_pd


def train_logistic_regression(X_train_pd, y_train, task='classification'):
    if task == 'classification':
        params = {
            'penalty': ['l1', 'l2', 'elasticnet', 'none'],
            'C': np.logspace(-4, 4, 20),
            'solver': ['lbfgs', 'newton-cg', 'liblinear', 'sag', 'saga'],
            'max_iter': [100, 1000, 2500, 5000]
        }
        try:
            model = RandomizedSearchCV(LogisticRegression(), param_distributions=params, cv=5, scoring='accuracy', error_score=np.nan)
            model.fit(X_train_pd, y_train)
            if model.best_estimator_ is None:
                raise ValueError("Model training failed. No best estimator found.")
            return model.best_estimator_
        except Exception as e:
            raise RuntimeError("Error occurred during model training:", e)

def train_linear_regression(X_train_pd, y_train):
        model = LinearRegression()
        model.fit(X_train_pd, y_train)
        return model

def train_decision_tree(X_train_pd, y_train, task='classification'):
    if task == 'classification':
        param_dist = {
            'criterion': ['gini', 'entropy'],
            'splitter': ['best', 'random'],
            'max_depth': [None, 5, 10, 15, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
        }
        random_search = RandomizedSearchCV(DecisionTreeClassifier(), param_distributions=param_dist, n_iter=10, cv=5, scoring='accuracy', random_state=42)
    elif task == 'regression':  # Regression
        param_dist = {
            'criterion': ['mse', 'friedman_mse', 'mae'],
            'splitter': ['best', 'random'],
            'max_depth': [None, 5, 10, 15, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
        }
        random_search = RandomizedSearchCV(DecisionTreeRegressor(), param_distributions=param_dist, n_iter=10, cv=5, scoring='r2', random_state=42)
    random_search.fit(X_train_pd, y_train)
    return random_search

def train_random_forest(X_train_pd, y_train, task='classification'):
    if task == 'classification':
        param_dist = {
            'n_estimators': randint(10, 200),
            'max_features': ['auto', 'sqrt', 'log2'],
            'max_depth': [None, 10, 20, 30, 40, 50],
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 20)
        }
        random_search = RandomizedSearchCV(RandomForestClassifier(), param_distributions=param_dist, n_iter=100, cv=5, scoring='accuracy', random_state=42)
    elif task == 'regression':  # Regression
        param_dist = {
            'n_estimators': randint(10, 200),
            'max_features': ['auto', 'sqrt', 'log2'],
            'max_depth': [None, 10, 20, 30, 40, 50],
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 20)
        }
        random_search = RandomizedSearchCV(RandomForestRegressor(), param_distributions=param_dist, n_iter=100, cv=5, scoring='r2', random_state=42)
    random_search.fit(X_train_pd, y_train)
    return random_search

def train_xg_boost(X_train_pd, y_train, task='classification'):
    if task == 'classification':
        param_dist = {
            'n_estimators': randint(50, 200),
            'learning_rate': [0.01, 0.1, 0.2, 0.3],
            'max_depth': randint(3, 10),
            'min_child_weight': randint(1, 10),
            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
            'gamma': [0, 0.1, 0.2, 0.3, 0.4],
        }
        random_search = RandomizedSearchCV(XGBClassifier(), param_distributions=param_dist, n_iter=100, cv=5, scoring='accuracy', random_state=42)
    elif task == 'regression': # Regression
        param_dist = {
            'n_estimators': randint(50, 200),
            'learning_rate': [0.01, 0.1, 0.2, 0.3],
            'max_depth': randint(3, 10),
            'min_child_weight': randint(1, 10),
            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
            'gamma': [0, 0.1, 0.2, 0.3, 0.4],
        }
        random_search = RandomizedSearchCV(XGBRegressor(),param_distributions=param_dist, n_iter=100, cv=5, scoring='accuracy', random_state=42)
        return random_search



# Main function to manage navigation between pages
def main():
    st.title("Machine Learning Model App")
    page = st.sidebar.selectbox("Menu", ["Home","Data Profiling", "Model Selection"])
    if page == "Home":
        st.write(home, unsafe_allow_html=True)
    if page == "Data Profiling":
        data_profiling()
    elif page == "Model Selection":
        df = load_dataset()
        model_selection(df)

if __name__ == "__main__":
    main()