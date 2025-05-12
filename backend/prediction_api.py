import os
import yaml
import glob
import pandas as pd
import xgboost as xgb
from src.logger import logger
from src.exception import CustomException
from src.data_preprocessing import DataPreprocessor

def load_xgb_model(model_base_path: str = "artifacts/runs") -> xgb.Booster:
    """
    Load the most recent XGBoost model saved in the given directory.

    :param model_base_path: Base directory where model runs are stored.
    :return: Loaded XGBoost Booster model.
    :raises CustomException: If model is missing or cannot be loaded.
    """
    try:
        run_dirs = sorted(
            [d for d in glob.glob(os.path.join(model_base_path, "*")) if os.path.isdir(d)],
            key=os.path.getmtime,
            reverse=True
        )

        if not run_dirs:
            raise CustomException("No saved model runs found.")

        latest_run_path = run_dirs[0]
        model_path = os.path.join(latest_run_path, "xgboost_model.json")

        if not os.path.exists(model_path):
            raise CustomException(f"No model found in latest run: {model_path}")

        logger.info(f"Loading model from: {model_path}")
        model = xgb.Booster()
        model.load_model(model_path)
        return model
    except Exception as e:
        logger.error("Failed to load latest model.", exc_info=True)
        raise CustomException("Error loading latest model", e)

def prediction(model, input_dict: dict, config_path: str = "config/config.yaml"):
    """
    Preprocesses the input and returns prediction and probability using the given model.

    :param model: Loaded XGBoost Booster model.
    :param input_dict: Raw input features as dictionary.
    :param config_path: Path to YAML config file for preprocessing.
    :return: Tuple (prediction, probability)
    """
    data = pd.DataFrame([input_dict])
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    preprocessor = DataPreprocessor(config)
    data = data[config.get("input_columns")]
    data = preprocessor.cast_categorical_columns(data)

    dmatrix = xgb.DMatrix(data, enable_categorical=True)
    prediction_probs = model.predict(dmatrix)
    predictions = (prediction_probs >= 0.5).astype(int)

    return predictions[0], prediction_probs[0]

def get_latest_prediction_file(directory: str = "daily_predictions") -> pd.DataFrame:
    """
    Loads the most recent CSV prediction file from the given directory.

    :param directory: Directory where daily prediction files are stored.
    :return: Pandas DataFrame of latest prediction file.
    """
    if not os.path.exists(directory) or not os.listdir(directory):
        return pd.DataFrame()

    files = sorted([f for f in os.listdir(directory) if f.endswith('.csv')])
    if not files:
        return pd.DataFrame()

    latest_file = os.path.join(directory, files[-1])
    df = pd.read_csv(latest_file)

    # Convert object columns to datetime if possible
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_datetime(df[col])
                df[col] = df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
            except Exception:
                continue

    return df
