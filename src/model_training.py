import xgboost as xgb
from scipy import stats
from src.logger import logger
from src.exception import CustomException
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score, log_loss, cohen_kappa_score
)

class ModelTraining:
    """
    Class responsible for training, hyperparameter tuning, and evaluating an XGBoost model.
    
    Attributes:
        config (dict): Configuration dictionary containing model and tuning parameters.
        X_train (pd.DataFrame): The training data (features).
        y_train (pd.Series): The training data (target variable).
        X_test (pd.DataFrame): The test data (features).
        y_test (pd.Series): The test data (target variable).
        eval_set (list): The evaluation set to monitor performance during training.
        model_cfg (dict): The configuration for the model (from `config`).
        tuning_cfg (dict): The configuration for hyperparameter tuning (from `config`).
    """

    def __init__(self, config: dict, X_train, y_train, X_test, y_test, eval_set):
        """
        Initializes the ModelTraining class with the configuration, training and testing data.
        
        :param config: Configuration settings for the model and hyperparameter tuning.
        :param X_train: Training data features.
        :param y_train: Training data target variable.
        :param X_test: Test data features.
        :param y_test: Test data target variable.
        :param eval_set: Evaluation set used for early stopping during model training.
        """
        try:
            # Initialize configuration and data
            self.config = config
            self.X_train = X_train
            self.y_train = y_train
            self.X_test = X_test
            self.y_test = y_test
            self.eval_set = eval_set
            self.model_cfg = config.get("model", {})
            self.tuning_cfg = config.get("tuning", {})
            logger.info("ModelTrainer initialized.")
        except Exception as e:
            logger.error("Failed to initialize ModelTrainer", exc_info=True)
            raise CustomException("ModelTrainer initialization error", e)

    def train_with_hyperparameter_tuning(self):
        """
        Performs hyperparameter tuning using RandomizedSearchCV and trains the XGBoost model.
        
        This method performs the following steps:
        1. Initializes the XGBoost classifier with base configuration.
        2. Performs RandomizedSearchCV to find the best hyperparameters.
        3. Trains the model with the best hyperparameters and returns the trained model.
        
        :return: The trained XGBoost model with the best hyperparameters.
        :raises CustomException: If hyperparameter tuning or model training fails.
        """
        try:
            logger.info("Starting hyperparameter tuning using RandomizedSearchCV...")
            
            # Initialize the base model
            base_model = xgb.XGBClassifier(**self.model_cfg)

            # Set up the RandomizedSearchCV for hyperparameter tuning
            param_search = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=self.tuning_cfg.get("param_distributions", {}),
                scoring=self.tuning_cfg.get("scoring", "roc_auc"),
                n_iter=self.tuning_cfg.get("n_iter", 100),
                cv=self.tuning_cfg.get("cv", 5),
                verbose=0,
                n_jobs=-1
            )

            # Fit the RandomizedSearchCV model
            param_search.fit(self.X_train, self.y_train, eval_set=self.eval_set, verbose=False)
            logger.info(f"Best hyperparameters: {param_search.best_params_}")

            # Train the best model with the tuned hyperparameters
            best_model = xgb.XGBClassifier(**self.model_cfg, **param_search.best_params_)
            best_model.fit(self.X_train, self.y_train, eval_set=self.eval_set, verbose=False)
            logger.info("Model training completed successfully.")
            
            return best_model
        except Exception as e:
            logger.error("Hyperparameter tuning and model training failed.", exc_info=True)
            raise CustomException("Training error", e)

    def evaluate(self, model):
        """
        Evaluates the model on both training and testing data and calculates various performance metrics.
        
        This method computes:
        - Accuracy
        - ROC AUC
        - Log Loss
        - F1-Score
        - Precision
        - Recall
        - Cohen Kappa
        
        It also logs confusion matrices for both training and testing sets.

        :param model: The trained model to be evaluated.
        :return: A dictionary containing the computed evaluation metrics.
        :raises CustomException: If evaluation fails.
        """
        try:
            logger.info("Evaluating model performance...")
            
            # Predictions and probabilities for both training and test datasets
            train_preds = model.predict(self.X_train)
            test_preds = model.predict(self.X_test)
            train_probs = model.predict_proba(self.X_train)[:, 1]
            test_probs = model.predict_proba(self.X_test)[:, 1]

            # Calculate various metrics
            metrics = {
                "train_accuracy": accuracy_score(self.y_train, train_preds),
                "test_accuracy": accuracy_score(self.y_test, test_preds),
                "train_roc_auc": roc_auc_score(self.y_train, train_probs),
                "test_roc_auc": roc_auc_score(self.y_test, test_probs),
                "train_log_loss": log_loss(self.y_train, train_probs),
                "test_log_loss": log_loss(self.y_test, test_probs),
                "f1_score": f1_score(self.y_test, test_preds),
                "precision": precision_score(self.y_test, test_preds),
                "recall": recall_score(self.y_test, test_preds),
                "train_kappa": cohen_kappa_score(self.y_train, train_preds, weights='quadratic'),
                "test_kappa": cohen_kappa_score(self.y_test, test_preds, weights='quadratic'),
            }

            # Log metrics
            for key, val in metrics.items():
                logger.info(f"{key}: {val:.4f}")

            # Log confusion matrices
            logger.info("Train Confusion Matrix:\n%s", confusion_matrix(self.y_train, train_preds))
            logger.info("Test Confusion Matrix:\n%s", confusion_matrix(self.y_test, test_preds))

            return metrics
        except Exception as e:
            logger.error("Model evaluation failed.", exc_info=True)
            raise CustomException("Evaluation error", e)

    def train_and_evaluate(self):
        """
        Trains the model using hyperparameter tuning and evaluates its performance.
        
        This method combines the training and evaluation processes into a single workflow:
        1. Trains the model with hyperparameter tuning.
        2. Evaluates the model on both training and test datasets.
        
        :return: The trained model and the evaluation metrics.
        :raises CustomException: If training or evaluation fails.
        """
        try:
            # Train the model with hyperparameter tuning
            model = self.train_with_hyperparameter_tuning()

            # Evaluate the trained model
            metrics = self.evaluate(model)

            return model, metrics
        except Exception as e:
            logger.error("Model training and evaluation failed.", exc_info=True)
            raise CustomException("Model training and evaluation failed", e)
