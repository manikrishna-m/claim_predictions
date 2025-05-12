import pytest
import pandas as pd
from src.data_ingestion import SyntheticDataIngestion
from src.exception import CustomException


@pytest.fixture
def synthetic_ingestion():
    """
    This fixture provides an instance of the SyntheticDataIngestion class
    with a predefined random seed (42) for generating synthetic data.
    """
    return SyntheticDataIngestion(random_seed=42)


def test_generate_synthetic_data_shape(synthetic_ingestion):
    """
    Test case to verify that the generated synthetic data has the correct shape.
    - Checks that the generated data is a DataFrame.
    - Verifies that the number of rows is equal to the specified value.
    - Ensures that the number of columns is greater than 0.
    """
    df = synthetic_ingestion._generate_synthetic_data(n_rows=500)
    
    # Assert that the generated data is a DataFrame
    assert isinstance(df, pd.DataFrame)
    
    # Assert that the generated data has the correct number of rows
    assert df.shape[0] == 500
    
    # Assert that the generated data has more than 0 columns
    assert df.shape[1] > 0  


def test_generate_synthetic_data_columns(synthetic_ingestion):
    """
    Test case to verify that the generated synthetic data contains all the expected columns.
    - Checks if all expected column names are present in the generated data.
    """
    df = synthetic_ingestion._generate_synthetic_data(n_rows=100)
    
    # List of expected columns in the synthetic data
    expected_columns = [
        'claim_status', 'age', 'height_cm', 'weight_kg', 'income',
        'financial_hist_1', 'financial_hist_2', 'financial_hist_3', 'financial_hist_4',
        'credit_score_1', 'credit_score_2', 'credit_score_3',
        'insurance_hist_1', 'insurance_hist_2', 'insurance_hist_3',
        'insurance_hist_4', 'insurance_hist_5', 'bmi', 'gender',
        'marital_status', 'occupation', 'location', 'prev_claim_rejected',
        'known_health_conditions', 'uk_residence',
        'family_history_1', 'family_history_2', 'family_history_3', 'family_history_4', 'family_history_5',
        'product_var_1', 'product_var_2', 'product_var_3', 'product_var_4',
        'health_status', 'driving_record', 'previous_claim_rate',
        'education_level', 'income_level', 'n_dependents', 'employment_type'
    ]
    
    # Assert that each of the expected columns exists in the generated DataFrame
    for col in expected_columns:
        assert col in df.columns


def test_collect_data_daily(synthetic_ingestion):
    """
    Test case to verify the loading of daily data.
    - Ensures that the data collected for the 'daily' mode has the correct number of rows (1200).
    - Verifies that the 'claim_status' column is present in the daily data.
    """
    df = synthetic_ingestion.collect_data(mode="daily")
    
    # Assert that the daily data has 1200 rows
    assert len(df) == 1200
    
    # Assert that the 'claim_status' column is in the daily data
    assert 'claim_status' in df.columns


def test_collect_data_monthly(synthetic_ingestion):
    """
    Test case to verify the loading of monthly data.
    - Ensures that the data collected for the 'monthly' mode has the correct number of rows (100000).
    - Verifies that the 'claim_status' column is present in the monthly data.
    """
    df = synthetic_ingestion.collect_data(mode="monthly")
    
    # Assert that the monthly data has 100000 rows
    assert len(df) == 100000
    
    # Assert that the 'claim_status' column is in the monthly data
    assert 'claim_status' in df.columns


def test_invalid_mode_falls_back_to_default(synthetic_ingestion):
    """
    Test case to verify that an invalid mode falls back to the default mode ('monthly').
    - When an invalid mode is passed, the method should still return data with 100000 rows (default).
    """
    df = synthetic_ingestion.collect_data(mode="invalid_mode")
    
    # Assert that the data returned in case of an invalid mode has 100000 rows (default mode)
    assert df.shape[0] == 100000
