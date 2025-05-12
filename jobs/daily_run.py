import os
import sys
import yaml
import glob
import mlflow
import pandas as pd
import xgboost as xgb
from datetime import datetime
from mlflow.tracking import MlflowClient

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.logger import logger
from src.exception import CustomException
from src.data_ingestion import SyntheticDataIngestion
from src.data_preprocessing import DataPreprocessor


class DailyPredictor:
    """
    Class responsible for loading daily data, preprocessing it, loading the latest trained model, 
    making predictions, and saving the results.

    Attributes:
        config (dict): Configuration dictionary loaded from the YAML file.
        model_name (str): The name of the ML model to use for predictions.
        output_dir (str): Directory where predictions will be saved.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initializes the DailyPredictor with the configuration file.

        :param config_path: Path to the configuration YAML file.
        """
        try:
            logger.info("Initializing DailyPredictor...")
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)

            self.model_name = self.config.get("mlflow", {}).get("model_name", "churn_model")
            self.output_dir = self.config.get("prediction_output_dir", "outputs")
            os.makedirs(self.output_dir, exist_ok=True)

            logger.info("DailyPredictor initialized successfully.")
        except Exception as e:
            logger.error("Failed to initialize DailyPredictor.", exc_info=True)
            raise CustomException("Initialization failed", e)

    def load_daily_data(self):
        """
        Loads the daily data using the SyntheticDataIngestion class.

        :return: DataFrame containing the daily data.
        :raises CustomException: If data loading fails.
        """
        try:
            logger.info("Loading daily data...")
            dataloader = SyntheticDataIngestion()
            data = dataloader.collect_data(mode="daily")
            logger.info("Daily data loaded.")
            return data
        except Exception as e:
            logger.error("Failed to load daily data.", exc_info=True)
            raise CustomException("Data loading failed", e)

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocesses the data (casts categorical columns).

        :param data: Raw input data to be processed.
        :return: Processed DataFrame with categorical columns casted.
        :raises CustomException: If data preprocessing fails.
        """
        try:
            logger.info("Preprocessing data...")
            preprocessor = DataPreprocessor(config=self.config)
            data = data[self.config.get("input_columns")]
            data = preprocessor.cast_categorical_columns(data)
            logger.info("Data preprocessing completed.")
            return data
        except Exception as e:
            logger.error("Data preprocessing failed.", exc_info=True)
            raise CustomException("Preprocessing failed", e)

    def load_xgb_model(self, model_base_path: str = "artifacts/runs") -> xgb.Booster:
        """
        Loads the latest XGBoost model from the specified directory.

        :param model_base_path: The base path to search for model runs.
        :return: The loaded XGBoost model.
        :raises CustomException: If model loading fails or no model is found.
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

    def make_predictions(self, model: xgb.Booster, data: pd.DataFrame) -> pd.DataFrame:
        """
        Makes predictions using the provided model and data.

        :param model: The trained XGBoost model to use for predictions.
        :param data: The preprocessed data for which predictions will be made.
        :return: DataFrame containing predictions and probabilities.
        :raises CustomException: If prediction fails.
        """
        try:
            logger.info("Making predictions...")
            target_column = self.config.get("target_column", "target")
            if target_column in data.columns:
                data = data.drop(columns=[target_column])

            dmatrix = xgb.DMatrix(data, enable_categorical=True)
            prediction_probs = model.predict(dmatrix)
            predictions = (prediction_probs >= 0.5).astype(int)

            results_df = data.copy()
            results_df["prediction"] = predictions
            results_df["probability"] = prediction_probs
            logger.info("Predictions completed.")
            return results_df
        except Exception as e:
            logger.error("Prediction step failed.", exc_info=True)
            raise CustomException("Prediction failed", e)

    def save_predictions(self, results_df: pd.DataFrame):
        """
        Saves the predictions to a CSV file.

        :param results_df: DataFrame containing the predictions to be saved.
        :raises CustomException: If saving predictions fails.
        """
        try:
            filename = f"predictions_{datetime.today().strftime('%Y-%m-%d')}.csv"
            output_path = os.path.join(self.output_dir, filename)
            results_df.to_csv(output_path, index=False)
            logger.info(f"Predictions saved to {output_path}")
        except Exception as e:
            logger.error("Failed to save predictions.", exc_info=True)
            raise CustomException("Saving predictions failed", e)

    def run(self):
        """
        Runs the entire daily prediction process: loading data, preprocessing, making predictions, 
        and saving the results.

        :raises CustomException: If any step in the process fails.
        """
        try:
            logger.info(f"--- Daily Prediction Run Started at {datetime.now()} ---")

            # Step 1: Load daily data
            data = self.load_daily_data()

            # Step 2: Preprocess data
            processed_data = self.preprocess_data(data)

            # Step 3: Load the latest model
            model = self.load_xgb_model()

            # Step 4: Make predictions
            results = self.make_predictions(model, processed_data)

            # Step 5: Save predictions
            self.save_predictions(results)

            logger.info(f"--- Daily Prediction Run Completed at {datetime.now()} ---")
        except Exception as e:
            logger.error("Daily prediction pipeline execution failed.", exc_info=True)
            raise CustomException("Daily prediction pipeline failed", e)


if __name__ == "__main__":
    predictor = DailyPredictor()
    predictor.run()
