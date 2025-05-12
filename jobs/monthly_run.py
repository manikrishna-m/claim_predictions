import os
import sys
import yaml
import pandas as pd
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.logger import logger
from src.exception import CustomException
from src.data_ingestion import SyntheticDataIngestion
from src.data_preprocessing import DataPreprocessor
from src.model_training import ModelTraining
from src.model_logging import LocalLogger

class InsuranceClaimsPipeline:
    """
    A class that represents the insurance claims prediction pipeline.
    It includes the following steps:
    - Loading data
    - Preprocessing data
    - Splitting data into train/test sets
    - Training and evaluating the model
    - Logging the model and metrics to MLflow

    Attributes:
        config (dict): Configuration loaded from the YAML file.
        data_loader (SyntheticDataIngestion): Data ingestion handler.
        preprocessor (DataPreprocessor): Data preprocessing handler.
        model (xgb.Booster): The trained model.
        metrics (dict): Performance metrics of the trained model.
        model_uri (str): URI for the MLflow model.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initializes the InsuranceClaimsPipeline by loading the configuration file.
        
        :param config_path: Path to the configuration file (default is "config/config.yaml").
        """
        try:
            logger.info("Initializing InsuranceClaimsPipeline...")
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)

            # Initialize components
            self.data_loader = SyntheticDataIngestion()
            self.preprocessor = None
            self.model = None
            self.metrics = None
            self.model_uri = f"models:/{self.config.get('mlflow', {}).get('model_name', 'claim_prediction')}/Production"

            logger.info("InsuranceClaimsPipeline initialized successfully.")
        except Exception as e:
            logger.error("Failed to initialize InsuranceClaimsPipeline.", exc_info=True)
            raise CustomException("Pipeline initialization failed", e)

    def load_monthly_data(self):
        """
        Loads the monthly data using the SyntheticDataIngestion class.

        :return: DataFrame containing the monthly data.
        :raises CustomException: If data loading fails.
        """
        try:
            logger.info("Loading monthly data...")
            data = self.data_loader.collect_data(mode="monthly")
            logger.info("Monthly data loaded successfully.")
            return data
        except Exception as e:
            logger.error("Failed to load monthly data.", exc_info=True)
            raise CustomException("Data loading failed", e)

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocesses the data by dropping columns with high missing values 
        and casting categorical columns.

        :param data: Raw input data to be processed.
        :return: Preprocessed DataFrame.
        :raises CustomException: If data preprocessing fails.
        """
        try:
            logger.info("Preprocessing data...")
            self.preprocessor = DataPreprocessor(self.config)

            # Drop columns with high missing values and cast categorical columns
            cleaned_data = self.preprocessor.drop_high_missing_columns(data)
            cleaned_data = self.preprocessor.cast_categorical_columns(cleaned_data)

            logger.info("Data preprocessing completed.")
            return cleaned_data
        except Exception as e:
            logger.error("Data preprocessing failed.", exc_info=True)
            raise CustomException("Preprocessing failed", e)

    def split_data(self, data: pd.DataFrame):
        """
        Splits the preprocessed data into train and test sets.

        :param data: Preprocessed data to be split.
        :return: X_train, X_test, y_train, y_test - Training and testing data and labels.
        :raises CustomException: If the train-test split fails.
        """
        try:
            logger.info("Splitting data into train and test sets...")
            X_train, X_test, y_train, y_test = self.preprocessor.split_train_test_data(data)
            logger.info(f"Train-test split completed. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logger.error("Failed to split data.", exc_info=True)
            raise CustomException("Train-test split failed", e)

    def train_and_evaluate_model(self, X_train, y_train, X_test, y_test, eval_set):
        """
        Trains the model using the training data and evaluates its performance on the test data.

        :param X_train: Feature data for training.
        :param y_train: Target labels for training.
        :param X_test: Feature data for testing.
        :param y_test: Target labels for testing.
        :param eval_set: Set of evaluation data to track model performance during training.
        :return: The trained model and its performance metrics.
        :raises CustomException: If model training or evaluation fails.
        """
        try:
            logger.info("Training model with hyperparameter tuning...")
            trainer = ModelTraining(self.config, X_train, y_train, X_test, y_test, eval_set)
            self.model, self.metrics = trainer.train_and_evaluate()
            logger.info("Model training completed.")
            return self.model, self.metrics
        except Exception as e:
            logger.error("Model training failed.", exc_info=True)
            raise CustomException("Model training failed", e)

    def log_to_mlflow(self):
        """
        Logs the trained model and its metrics to MLflow.

        :raises CustomException: If logging to MLflow fails.
        """
        try:
            logger.info("Logging model and metrics to MLflow...")
            mlflow_logger = LocalLogger(self.config, self.model, self.metrics)
            mlflow_logger.log()
            logger.info("Logged to MLflow successfully.")
        except Exception as e:
            logger.error("MLflow logging failed.", exc_info=True)
            raise CustomException("MLflow logging failed", e)

    def run(self):
        """
        Runs the entire insurance claims prediction pipeline:
        - Load monthly data
        - Preprocess data
        - Split data into train/test
        - Train the model and evaluate it
        - Log model and metrics to MLflow

        :raises CustomException: If any part of the pipeline fails.
        """
        try:
            logger.info(f"--- Monthly Retraining Pipeline Started at {datetime.now()} ---")

            # Step 1: Load monthly data
            data = self.load_monthly_data()

            # Step 2: Preprocess the data
            processed_data = self.preprocess_data(data)

            # Step 3: Split the data into training and testing sets
            X_train, X_test, y_train, y_test = self.split_data(processed_data)

            # Step 4: Set evaluation set
            eval_set = [(X_test, y_test)]

            # Step 5: Train and evaluate the model
            self.model, self.metrics = self.train_and_evaluate_model(X_train, y_train, X_test, y_test, eval_set)

            # Step 6: Log model and metrics to MLflow
            self.log_to_mlflow()

            logger.info(f"--- Monthly Retraining Pipeline Completed at {datetime.now()} ---")

        except Exception as e:
            logger.error("Monthly retraining pipeline execution failed.", exc_info=True)
            raise CustomException("Monthly retraining pipeline execution failed", e)

if __name__ == "__main__":
    pipeline = InsuranceClaimsPipeline()
    pipeline.run()
