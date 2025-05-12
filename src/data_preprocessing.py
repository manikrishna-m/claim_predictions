import yaml
import pandas as pd
from typing import Optional, Tuple
from src.logger import logger
from src.exception import CustomException
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    """
    A class responsible for preprocessing data by handling missing values, 
    categorical columns, and splitting the dataset into training and testing sets.
    """

    def __init__(self, config: dict, config_path: str = "config/config.yaml"):
        """
        Initializes the DataPreprocessor class, loading the configuration and 
        setting up the necessary preprocessing steps.

        :param config: A dictionary containing configuration parameters for preprocessing.
        :param config_path: The path to the YAML configuration file (default is "config/config.yaml").
        :raises CustomException: If initialization fails.
        """
        try:
            self.config = config
            self.input_columns = config.get("input_columns")
            self.target_column = config.get("target_column", "claim_status")

            if self.input_columns is None:
                raise ValueError("Missing 'input_columns' in config.")

            self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
            self.missing_summary: Optional[pd.DataFrame] = None

            logger.info("DataPreprocessor initialized successfully.")
        except Exception as e:
            logger.error("Failed to initialize DataPreprocessor", exc_info=True)
            raise CustomException("Data preprocessing initialization error", e)

    def summarize_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Summarizes the missing values in the provided DataFrame, including the 
        total count of missing values, percentage of missing values, and data types 
        of each column.

        :param data: The DataFrame for which missing values are to be summarized.
        :returns: A DataFrame summarizing the missing values in each column.
        :raises CustomException: If the operation fails.
        """
        try:
            total_missing = data.isnull().sum()
            percent_missing = (total_missing / data.shape[0]) * 100
            types = data.dtypes.astype(str)

            self.missing_summary = pd.DataFrame({
                'Total': total_missing,
                'Percent': percent_missing,
                'Types': types
            }).sort_values(by='Total', ascending=False)

            logger.info("Missing value summary generated.")
            return self.missing_summary
        except Exception as e:
            logger.error("Failed to summarize missing values.", exc_info=True)
            raise CustomException("Error in summarize_missing_values", e)

    def drop_high_missing_columns(self, data: pd.DataFrame, threshold: float = 50.0) -> pd.DataFrame:
        """
        Drops columns from the DataFrame that have a percentage of missing values greater than 
        the specified threshold.

        :param data: The DataFrame from which columns with high missing values will be dropped.
        :param threshold: The threshold percentage for dropping columns (default is 50%).
        :returns: The cleaned DataFrame with high-missing columns removed.
        :raises CustomException: If the operation fails.
        """
        try:
            if self.missing_summary is None:
                self.summarize_missing_values(data)

            cols_to_drop = self.missing_summary[self.missing_summary['Percent'] > threshold].index.tolist()
            cleaned_data = data.drop(columns=cols_to_drop)
            logger.info(f"Dropped columns with >{threshold}% missing values: {cols_to_drop}")
            return cleaned_data
        except Exception as e:
            logger.error("Failed to drop high-missing columns.", exc_info=True)
            raise CustomException("Error in drop_high_missing_columns", e)

    def cast_categorical_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Casts the columns specified in the configuration as categorical columns.

        :param data: The DataFrame in which categorical columns are to be cast.
        :returns: The DataFrame with categorical columns cast to the 'category' dtype.
        :raises CustomException: If the operation fails.
        """
        try:
            categorical_columns = self.config.get("categorical_columns", [])
            for col in categorical_columns:
                if col in data.columns:
                    data[col] = data[col].astype("category")
            logger.info("Categorical columns cast to 'category' dtype.")
            return data
        except Exception as e:
            logger.error("Failed to cast categorical columns.", exc_info=True)
            raise CustomException("Error in cast_categorical_columns", e)

    def split_train_test_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Splits the data into training and testing sets based on the provided configuration.

        :param data: The DataFrame to be split into train and test sets.
        :returns: A tuple containing the training and testing features and target variables.
        :raises CustomException: If the operation fails.
        """
        try:
            self.X = data[self.input_columns]
            self.y = data[self.target_column]

            split_cfg = self.config.get("split", {})
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X,
                self.y,
                test_size=split_cfg.get("test_size", 0.2),
                stratify=self.y if split_cfg.get("stratify", True) else None,
                random_state=split_cfg.get("random_state", 1889)
            )
            logger.info("Train-test split completed.")
            return self.X_train, self.X_test, self.y_train, self.y_test
        except Exception as e:
            logger.error("Failed to perform train-test split.", exc_info=True)
            raise CustomException("Error in split_train_test_data", e)
