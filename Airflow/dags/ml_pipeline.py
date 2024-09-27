from airflow import DAG
import os
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
from datetime import datetime
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pickle
import json

# Define paths
DATA_PATH = 'Airflow/data/iris_data.csv'
MODEL_PATH = 'Airflow/data/decision_tree_model.pkl'

# Load data
def load_data():
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)

    iris = load_iris()
    data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    data['target'] = iris.target
    data.to_csv(DATA_PATH, index=False) 
    return DATA_PATH  # Return the path of the CSV file

# Train the model
def train_model(data_path, **kwargs):
    data = pd.read_csv(data_path)  # Read data from the CSV file
    X = data.drop('target', axis=1)
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    
    # Save the model using pickle
    with open(MODEL_PATH, 'wb') as model_file:
        pickle.dump(model, model_file)

    # Debugging: Print the shapes of the test data
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    # Push test data to XCom
    ti = kwargs['ti']
    ti.xcom_push(key='X_test', value=json.dumps(X_test.values.tolist()))  # Convert to JSON serializable format
    ti.xcom_push(key='y_test', value=json.dumps(y_test.tolist()))  # Convert to JSON serializable format

    return MODEL_PATH

# Evaluate the model
def evaluate_model(ti):
    """
    Evaluate the machine learning model using the test data.
    """

    # Pulling test data from XCom
    X_test = ti.xcom_pull(task_ids='train_model', key='X_test')
    y_test = ti.xcom_pull(task_ids='train_model', key='y_test')

    # Debug log to inspect what X_test is before parsing
    print(f"Value of X_test before JSON parsing: {X_test}")

    # Check if X_test or y_test is empty or None before attempting to parse
    if not X_test or not y_test:
        raise ValueError("X_test or y_test is empty or None. Cannot proceed with JSON parsing.")

    # Parse the test data
    try:
        X_test = json.loads(X_test)
        y_test = json.loads(y_test)
    except json.JSONDecodeError as e:
        print(f"JSON decoding failed: {e}")
        raise

    # Load the trained model
    with open(MODEL_PATH, 'rb') as model_file:
        model = pickle.load(model_file)

    # Convert X_test and y_test back to DataFrame and Series
    X_test = pd.DataFrame(X_test, columns=load_iris().feature_names)
    y_test = pd.Series(y_test)

    # Evaluate the model
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model accuracy: {accuracy}")

# Define DAG and its default arguments
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 9, 1),
    'retries': 1
}

dag = DAG(
    dag_id='ml_pipeline', 
    default_args=default_args, 
    description='Machine Learning Pipeline', 
    schedule_interval='@daily'
)

# Create tasks for each function
start_task = DummyOperator(task_id='start', dag=dag)

load_data_task = PythonOperator(
    task_id='load_data',
    python_callable=load_data,
    dag=dag
)

train_model_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    op_kwargs={'data_path': '{{ task_instance.xcom_pull(task_ids="load_data") }}'},  # Pass the path of the CSV file
    provide_context=True,  # Added to pass context for XCom usage
    dag=dag
)

evaluate_model_task = PythonOperator(
    task_id='evaluate_model',
    python_callable=evaluate_model,
    provide_context=True,  # This line allows passing the task_instance to access XCom
    dag=dag
)

end_task = DummyOperator(task_id='end', dag=dag)

# Set task dependencies
start_task >> load_data_task >> train_model_task >> evaluate_model_task >> end_task
