import sys
import time
import string
import hashlib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_classification

from src.logger import logger
from src.exception import CustomException

class DataIngestion:
    """
    Base class for data ingestion. Defines the interface for collecting data.
    """

    def __init__(self, random_seed: int = 1889):
        """
        Initializes the DataIngestion object with a random seed for reproducibility.

        :param random_seed: Random seed for reproducibility (default is 1889).
        """
        self.random_seed = random_seed
        np.random.seed(self.random_seed)

    def collect_data(self, query: str = None, mode: str = "daily") -> pd.DataFrame:
        """
        Abstract method that must be implemented by subclasses to collect data.
        This method is designed to be overridden by subclasses that provide
        their own implementation of data collection.

        :param query: Query string for fetching data (default is None).
        :param mode: Mode of data collection (default is "daily").
        :raises NotImplementedError: If not overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class DatabaseIngestion(DataIngestion):
    """
    Concrete subclass of DataIngestion to collect data from a database.
    Implements the data collection logic for fetching data from a database.
    """

    def collect_data(self, query: str, mode: str = "daily") -> pd.DataFrame:
        """
        Collects data from the database using the given query and mode.
        
        :param query: SQL query for fetching data from the database.
        :param mode: Mode of data collection (default is "daily").
        :raises CustomException: If an error occurs while fetching data from the database.
        """
        try:
            logger.info(f"Fetching data from database with query: {query}")
            # Implement actual DB fetching logic here
            raise NotImplementedError("Database connection not yet implemented.")
        except Exception as e:
            raise CustomException("Error while collecting data from database", sys) from e


class SyntheticDataIngestion(DataIngestion):
    """
    Concrete subclass of DataIngestion to generate synthetic data for testing.
    Implements the logic for simulating synthetic data based on the given mode.
    """

    def collect_data(self, query: str = None, mode: str = "daily") -> pd.DataFrame:
        """
        Collects synthetic data based on the mode (either 'daily' or 'bulk').
        Simulates the data using random numbers and predefined distributions.

        :param query: Not used for synthetic data ingestion.
        :param mode: Mode of data collection, either "daily" or "bulk".
        :returns: A DataFrame containing simulated data.
        :raises CustomException: If an error occurs while generating synthetic data.
        """
        try:
            n_rows = 1200 if mode == "daily" else 100000
            if mode == "daily":
                today = datetime.today().strftime('%Y-%m-%d')
                self.random_seed = int(hashlib.sha256(today.encode('utf-8')).hexdigest(), 16) % (10 ** 8)
            else:
                self.random_seed = 1889  # Reset the random seed for bulk data
                
            logger.info(f"Simulating {n_rows} rows of synthetic data ({mode} mode)...")
            return self._generate_synthetic_data(n_rows=n_rows)
        
        except Exception as e:
            raise CustomException("Error while collecting synthetic data", sys) from e

    def _generate_synthetic_data(self, n_rows: int) -> pd.DataFrame:
        """
        Generates synthetic data based on random distributions and scaling.

        :param n_rows: The number of rows to generate in the synthetic dataset.
        :returns: A DataFrame containing the simulated data.
        :raises CustomException: If an error occurs during data generation.
        """
        try:
            # Number of features and their configuration
            n_features = 16
            features, labels = make_classification(
                n_samples=n_rows,
                n_features=n_features,
                n_informative=7,
                n_redundant=4,
                n_repeated=3,
                n_classes=2,
                class_sep=1.2,
                flip_y=0.035,
                weights=[0.85, 0.15],
                random_state=self.random_seed,
            )

            # Creating a DataFrame from the generated features
            df = pd.DataFrame(features, columns=[f'numeric_{i+1}' for i in range(n_features)])
            df.insert(value=labels, loc=0, column='claim_status')

            # Renaming columns to more meaningful names
            rename_map = {
                'numeric_1': 'age',
                'numeric_2': 'height_cm',
                'numeric_3': 'weight_kg',
                'numeric_4': 'income',
                'numeric_5': 'financial_hist_1',
                'numeric_6': 'financial_hist_2',
                'numeric_7': 'financial_hist_3',
                'numeric_8': 'financial_hist_4',
                'numeric_9': 'credit_score_1',
                'numeric_10': 'credit_score_2',
                'numeric_11': 'credit_score_3',
                'numeric_12': 'insurance_hist_1',
                'numeric_13': 'insurance_hist_2',
                'numeric_14': 'insurance_hist_3',
                'numeric_15': 'insurance_hist_4',
                'numeric_16': 'insurance_hist_5',
            }
            df = df.rename(columns=rename_map)

            # Scaling numeric features to specified ranges
            df['age'] = MinMaxScaler((18, 95)).fit_transform(df['age'].values[:, None]).astype(int)
            df['height_cm'] = MinMaxScaler((140, 210)).fit_transform(df['height_cm'].values[:, None]).astype(int)
            df['weight_kg'] = MinMaxScaler((45, 125)).fit_transform(df['weight_kg'].values[:, None]).astype(int)
            df['income'] = MinMaxScaler((0, 250_000)).fit_transform(df['income'].values[:, None]).astype(int)
            df['credit_score_1'] = MinMaxScaler((0, 999)).fit_transform(df['credit_score_1'].values[:, None]).astype(int)
            df['credit_score_2'] = MinMaxScaler((0, 700)).fit_transform(df['credit_score_2'].values[:, None]).astype(int)
            df['credit_score_3'] = MinMaxScaler((0, 710)).fit_transform(df['credit_score_3'].values[:, None]).astype(int)
            df['bmi'] = (df['weight_kg'] / ((df['height_cm'] / 100) ** 2)).astype(int)

            # Adding other categorical and random data columns
            df['gender'] = np.where(df['claim_status'] == 0,
                                    np.random.choice([1, 0], size=n_rows, p=[0.46, 0.54]),
                                    np.random.choice([1, 0], size=n_rows, p=[0.52, 0.48]))

            df['marital_status'] = np.random.choice(list("ABCDEF"), size=n_rows, p=[0.2, 0.15, 0.1, 0.25, 0.15, 0.15])
            df['occupation'] = np.random.choice(list("ABCDEFG"), size=n_rows)
            df['location'] = np.random.choice(list(string.ascii_uppercase), size=n_rows)

            df['prev_claim_rejected'] = np.where(df['claim_status'] == 0,
                                                 np.random.choice([1, 0], size=n_rows, p=[0.08, 0.92]),
                                                 np.random.choice([1, 0], size=n_rows, p=[0.16, 0.84]))

            df['known_health_conditions'] = np.random.choice([1, 0], size=n_rows, p=[0.06, 0.94])
            df['uk_residence'] = np.random.choice([1, 0], size=n_rows, p=[0.76, 0.24])

            # Simulating family history and other binary columns
            for i in range(1, 6):
                col = f'family_history_{i}'
                df[col] = np.random.choice([1, 0] if i != 3 else [1, None, 0],
                                           size=n_rows,
                                           p=[0.2 + 0.02*i, 0.8 - 0.02*i] if i != 3 else [0.12, 0.81, 0.07])

            # Adding product variables and health-related columns
            for i in range(1, 3):
                df[f'product_var_{i}'] = np.random.choice([1, 0], size=n_rows, p=[0.4 + 0.15*i, 0.6 - 0.15*i])

            df['product_var_3'] = np.random.choice(list("ABCD"), size=n_rows, p=[0.23, 0.28, 0.31, 0.18])
            df['product_var_4'] = np.random.choice([1, 0], size=n_rows, p=[0.76, 0.24])
            df['health_status'] = np.random.randint(1, 5, size=n_rows)
            df['driving_record'] = np.random.randint(1, 5, size=n_rows)

            # Previous claims rate and other demographic info
            df['previous_claim_rate'] = np.where(df['claim_status'] == 0,
                                                 np.random.choice(range(1, 6), size=n_rows, p=[0.48, 0.29, 0.12, 0.08, 0.03]),
                                                 np.random.choice(range(1, 6), size=n_rows, p=[0.12, 0.28, 0.34, 0.19, 0.07]))

            df['education_level'] = np.random.randint(0, 7, size=n_rows)
            df['income_level'] = pd.cut(df['income'], bins=5, labels=False, include_lowest=True)
            df['n_dependents'] = np.random.choice(range(1, 6), size=n_rows, p=[0.23, 0.32, 0.27, 0.11, 0.07])
            df['employment_type'] = np.random.choice([1, None, 0], size=n_rows, p=[0.16, 0.7, 0.14])

            logger.info("Synthetic data generated successfully.")
            return df

        except Exception as e:
            raise CustomException("Error while generating synthetic data", sys) from e
