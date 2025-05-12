import os
import pandas as pd
from datetime import datetime
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import FastAPI, Request, HTTPException

from src.logger import logger
from backend.prediction_api import get_latest_prediction_file, load_xgb_model, prediction

# Initialize FastAPI app with title and description
app = FastAPI(
    title="HISCOX Prediction Dashboard",
    description="API for insurance prediction dashboard"
)

# Configure Jinja2 for rendering HTML templates
templates = Jinja2Templates(directory="templates")

# Global model variable to hold the loaded ML model
model = None

@app.on_event("startup")
def load_model_on_startup():
    """
    Load the trained XGBoost model into memory at application startup.
    This ensures it's available for prediction without reloading every time.
    """
    global model
    model = load_xgb_model()
    logger.info("Model loaded on startup.")

# Define the expected input schema using Pydantic BaseModel
class PredictionInput(BaseModel):
    age: int
    height_cm: int
    weight_kg: int
    income: int
    financial_hist_1: float
    financial_hist_2: float
    financial_hist_3: float
    financial_hist_4: float
    credit_score_1: int
    credit_score_2: int
    credit_score_3: int
    insurance_hist_1: float
    insurance_hist_2: float
    insurance_hist_3: float
    insurance_hist_4: float
    insurance_hist_5: float
    bmi: int
    gender: int
    marital_status: str
    occupation: str
    location: str
    prev_claim_rejected: int
    known_health_conditions: int
    uk_residence: int
    family_history_1: int
    family_history_2: int
    family_history_3: str
    family_history_4: int
    family_history_5: int
    product_var_1: int
    product_var_2: int
    product_var_3: str
    product_var_4: int
    health_status: int
    driving_record: int
    previous_claim_rate: int
    education_level: int
    income_level: int
    n_dependents: int
    employment_type: str

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """
    Render the main dashboard HTML page.

    Args:
        request (Request): The incoming HTTP request object.

    Returns:
        HTMLResponse: Rendered dashboard page.
    """
    try:
        return templates.TemplateResponse("index.html", {"request": request})
    except Exception as e:
        logger.error(f"Error loading dashboard: {e}")
        raise HTTPException(status_code=500, detail="Error loading dashboard")

@app.get("/api/daily_predictions")
async def get_daily_predictions():
    """
    Retrieve the latest daily predictions from saved CSV files.

    Returns:
        List[Dict]: A list of prediction records in dictionary format.
    """
    try:
        df = get_latest_prediction_file("daily_predictions")
        return df.to_dict(orient="records") if not df.empty else []
    except Exception as e:
        logger.error(f"Error loading prediction data: {e}")
        raise HTTPException(status_code=500, detail="Error loading prediction data")

@app.post("/api/manual_predict")
async def make_manual_prediction(input_data: PredictionInput):
    """
    Perform a manual prediction using input features provided by the user.

    Args:
        input_data (PredictionInput): Input features for the prediction.

    Returns:
        Dict: Contains prediction label and probability score.
    """
    try:
        input_dict = input_data.dict()
        prediction_label, probability = prediction(model, input_dict)

        return {
            "prediction": int(prediction_label),     # Convert to int if model returns np.int64
            "probability": float(probability)         # Convert to float if model returns np.float64
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=f"Error making prediction: {str(e)}")

@app.get("/api/system/health")
async def health_check():
    """
    Perform a health check of the application.

    Returns:
        Dict: System status, model status, and prediction file count.
    """
    try:
        return {
            "status": "healthy" if model else "degraded",
            "timestamp": datetime.now().isoformat(),
            "model_loaded": model is not None,
            "model_version": model.__dict__.get('_model_meta', {}).get('run_id', 'unknown') if model else None,
            "prediction_files": len(os.listdir("daily_predictions")) if os.path.exists("daily_predictions") else 0
        }
    except Exception as e:
        logger.error(f"Health check error: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")
