import joblib
import os

BASE_DIR = r"C:\Users\sanjy\Downloads\Employee_Attrition_Prediction"
PERF_FILE = os.path.join(BASE_DIR, "best_models.pkl")

def save_performance(model_name, performance_dict, filename=PERF_FILE):
    """
    Saves model performance to a shared dictionary.
    
    Parameters:
    - model_name: str, name of the model
    - performance_dict: dict, contains metrics like accuracy, precision, recall, f1 score
    - filename: str, file to save the dictionary
    """
    if os.path.exists(filename):
        best_models = joblib.load(filename)
    else:
        best_models = {}

    best_models[model_name] = performance_dict
    joblib.dump(best_models, filename)

    print(f"Saved performance for {model_name}")

def load_performance(filename="best_models.pkl"):
    """
    Loads the saved performance dictionary.
    """
    if os.path.exists(filename):
        return joblib.load(filename)
    else:
        return {}
