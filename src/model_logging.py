import os
import json
import tempfile
from typing import Any, Dict, Optional
from src.logger import logger
from src.exception import CustomException


class LocalLogger:
    """
    A class that handles logging of model parameters, metrics, and the model itself locally.
    It saves the logs in the 'artifacts' directory under a specific run folder.

    Attributes:
        config (dict): Configuration settings for logging.
        model (Any): The trained model to be saved.
        metrics (dict): The model's performance metrics.
        params (dict, optional): Model parameters, default is an empty dictionary.
        run_name (str, optional): The name of the current run, default is "default_run".
        output_dir (str): The directory where logs and the model will be saved.
    """

    def __init__(
        self,
        config: dict,
        model: Any,
        metrics: Dict[str, float],
        params: Optional[Dict[str, Any]] = None,
        run_name: Optional[str] = None
    ):
        """
        Initializes the LocalLogger with the given configuration, model, metrics, and parameters.
        Creates the output directory where the logs and model will be stored.

        :param config: Configuration settings for logging.
        :param model: The trained model to be saved.
        :param metrics: A dictionary containing the model's performance metrics.
        :param params: Model parameters, default is an empty dictionary.
        :param run_name: The name of the current run, default is "default_run".
        """
        try:
            self.config = config
            self.model = model
            self.metrics = metrics
            self.params = params or {}
            self.run_name = run_name or "default_run"
            
            # Define the output directory and create it if it doesn't exist
            self.output_dir = os.path.join("artifacts", "runs", self.run_name)
            os.makedirs(self.output_dir, exist_ok=True)
            logger.info("LocalLogger initialized.")
        except Exception as e:
            logger.error("Failed to initialize LocalLogger", exc_info=True)
            raise CustomException("LocalLogger initialization error", e)

    def log(self):
        """
        Saves the model, parameters, and metrics to local disk.

        - Saves the model as a JSON file.
        - Saves the model's parameters as a JSON file.
        - Saves the metrics as a JSON file.

        Raises:
            CustomException: If there is any error while logging the model, parameters, or metrics.
        """
        try:
            logger.info("Saving model, parameters, and metrics locally...")

            # Save parameters to a JSON file
            params_path = os.path.join(self.output_dir, "params.json")
            with open(params_path, "w") as f:
                json.dump(self.params, f, indent=4)
            logger.info(f"Parameters saved to {params_path}")

            # Save metrics to a JSON file
            metrics_path = os.path.join(self.output_dir, "metrics.json")
            with open(metrics_path, "w") as f:
                json.dump(self.metrics, f, indent=4)
            logger.info(f"Metrics saved to {metrics_path}")

            # Save the model to a JSON file
            model_path = os.path.join(self.output_dir, "xgboost_model.json")
            try:
                # Try saving with the booster if available
                self.model.get_booster().save_model(model_path)
                logger.info(f"Model saved to {model_path}")
            except Exception as save_error:
                logger.warning(f"Error saving with booster: {save_error}")
                try:
                    # Fallback to saving the model directly if booster is not available
                    self.model.save_model(model_path)
                    logger.info(f"Model saved to {model_path}")
                except Exception as fallback_error:
                    logger.error(f"Failed to save model: {fallback_error}")
                    raise CustomException("Model saving error", fallback_error)

        except Exception as e:
            logger.error("Failed to log locally.", exc_info=True)
            raise CustomException(f"Local logging error: {str(e)}")


# import mlflow
# import mlflow.xgboost
# from src.logger import logger
# from src.exception import CustomException
# from typing import Any, Dict, Optional
# import os
# import tempfile
# from mlflow.tracking import MlflowClient


# class MLflowLogger:
#     def __init__(
#         self, 
#         config: dict, 
#         model: Any, 
#         metrics: Dict[str, float], 
#         params: Optional[Dict[str, Any]] = None,
#         run_name: Optional[str] = None
#     ):
#         try:
#             self.config = config
#             self.model = model
#             self.metrics = metrics
#             self.params = params or {}
#             self.run_name = run_name
#             self.experiment_name = config.get("mlflow", {}).get("experiment_name", "default_experiment")
#             self.model_name = config.get("mlflow", {}).get("model_name", "model")
#             logger.info("MLflowLogger initialized.")
#         except Exception as e:
#             logger.error("Failed to initialize MLflowLogger", exc_info=True)
#             raise CustomException("MLflowLogger initialization error", e)

#     def log(self):
#         """Logs the model, parameters, and metrics to MLflow."""
#         try:
#             logger.info("Logging model, parameters, and metrics to MLflow...")
#             mlflow.set_experiment(self.experiment_name)

#             with mlflow.start_run(run_name=self.run_name):
#                 # Log parameters
#                 for param_name, value in self.params.items():
#                     mlflow.log_param(param_name, value)

#                 # Log metrics
#                 for metric_name, value in self.metrics.items():
#                     mlflow.log_metric(metric_name, value)

#                 # Use temporary directory for model saving
#                 with tempfile.TemporaryDirectory() as tmp_dir:
#                     # Save model with JSON format for categorical features
#                     # The error indicates we need to save in JSON format for categorical splits
#                     model_path = os.path.join(tmp_dir, "model.json")
                    
#                     try:
#                         # For models with categorical features, we need to save in JSON format
#                         self.model.get_booster().save_model(model_path)
#                     except Exception as save_error:
#                         logger.warning(f"Error saving model with default method: {str(save_error)}")
                        
#                         # Try explicitly with 'json' format parameter if available
#                         try:
#                             self.model.save_model(model_path)
#                         except Exception:
#                             # Last resort, try pickle serialization with MLflow
#                             logger.warning("Falling back to MLflow's built-in serialization")
#                             mlflow.xgboost.log_model(
#                                 self.model,
#                                 artifact_path="xgboost-model",
#                                 registered_model_name=self.model_name,
#                                 # Specify serialization format explicitly
#                                 save_format="json"
#                             )
#                             return  # Skip the rest since we've already logged the model
                    
#                     # If we got here, we've successfully saved the model to a file
#                     logger.info(f"Model saved successfully to {model_path}")
                    
#                     # Log the model artifact
#                     mlflow.log_artifact(model_path, artifact_path="model")
                    
#                     # Register the model
#                     model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
#                     mlflow.register_model(model_uri, self.model_name)

#                 # Transition model stages
#                 try:
#                     client = MlflowClient()
#                     latest_versions = client.get_latest_versions(self.model_name, stages=["None"])
                    
#                     if latest_versions:
#                         model_version = latest_versions[0]
                        
#                         client.transition_model_version_stage(
#                             name=self.model_name,
#                             version=model_version.version,
#                             stage="Staging"
#                         )
#                         logger.info(f"Model version {model_version.version} moved to Staging")
                        
#                         client.transition_model_version_stage(
#                             name=self.model_name,
#                             version=model_version.version,
#                             stage="Production"
#                         )
#                         logger.info(f"Model version {model_version.version} moved to Production")
#                 except Exception as e:
#                     logger.warning(f"Failed to transition model stages: {str(e)}")

#         except Exception as e:
#             logger.error("Failed to log to MLflow.", exc_info=True)
#             raise CustomException(f"MLflow logging error: {str(e)}")