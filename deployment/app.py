import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from typing import List, Dict
import uvicorn
from datetime import datetime


app = FastAPI(
    title="Loan Default Prediction API",
    description="Production API for loan default prediction",
    version="1.0.0"
)


class LoanApplication(BaseModel):
    """Loan application input schema"""
    Client_Income: float
    Car_Owned: int
    Bike_Owned: int
    Active_Loan: int
    House_Own: int
    Child_Count: int
    Credit_Amount: float
    Loan_Annuity: float
    Accompany_Client: str = None
    Client_Income_Type: str = None
    Client_Education: str = None
    Client_Marital_Status: str = None
    Client_Gender: str = None
    Loan_Contract_Type: str = None
    Client_Housing_Type: str = None
    Population_Region_Relative: float = None
    Age_Days: float
    Employed_Days: float
    Registration_Days: float = None
    ID_Days: float = None
    Own_House_Age: float = None
    Mobile_Tag: int = None
    Homephone_Tag: int = None
    Workphone_Working: int = None
    Client_Occupation: str = None
    Client_Family_Members: int = None
    Cleint_City_Rating: int = None
    Application_Process_Day: int = None
    Application_Process_Hour: int = None
    Client_Permanent_Match_Tag: int = None
    Client_Contact_Work_Tag: int = None
    Type_Organization: str = None
    Score_Source_1: float = None
    Score_Source_2: float = None
    Score_Source_3: float = None
    Social_Circle_Default: int = None
    Phone_Change: float = None
    Credit_Bureau: int = None


class PredictionResponse(BaseModel):
    """Prediction response schema"""
    application_id: str
    default_probability: float
    prediction: str
    risk_category: str
    timestamp: str


class ModelArtifacts:
    """Load and cache model artifacts"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelArtifacts, cls).__new__(cls)
            cls._instance.load_artifacts()
        return cls._instance
    
    def load_artifacts(self):
        """Load all model artifacts"""
        try:
            self.model = joblib.load('models/best_model_xgboost_tuned.pkl')
            self.preprocessor = joblib.load('models/preprocessor.pkl')
            self.feature_engineer = joblib.load('models/feature_engineer.pkl')
            self.feature_names = joblib.load('models/feature_names.pkl')
            
            import json
            with open('models/model_metadata.json', 'r') as f:
                self.metadata = json.load(f)
            
            print("Model artifacts loaded successfully")
        except Exception as e:
            print(f"Error loading model artifacts: {str(e)}")
            raise


artifacts = ModelArtifacts()


@app.get("/")
def root():
    """Root endpoint"""
    return {
        "message": "Loan Default Prediction API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": artifacts.model is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/model/info")
def model_info():
    """Get model information"""
    return {
        "model_name": artifacts.metadata.get('model_name', 'Unknown'),
        "num_features": len(artifacts.metadata.get('features', [])),
        "optimal_threshold": artifacts.metadata.get('optimal_threshold', 0.5),
        "test_performance": artifacts.metadata.get('test_performance', {})
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(loan: LoanApplication):
    """Make prediction for a single loan application"""
    try:
        input_data = pd.DataFrame([loan.dict()])
        
        input_data = artifacts.preprocessor.preprocess_pipeline(
            input_data, fit=False
        )
        
        input_data = artifacts.feature_engineer.feature_engineering_pipeline(
            input_data, fit=False
        )
        
        available_features = [f for f in artifacts.feature_names if f in input_data.columns]
        X = input_data[available_features]
        
        for missing_feature in set(artifacts.feature_names) - set(available_features):
            X[missing_feature] = 0
        
        X = X[artifacts.feature_names]
        
        probability = artifacts.model.predict_proba(X)[0, 1]
        threshold = artifacts.metadata.get('optimal_threshold', 0.5)
        prediction = int(probability >= threshold)
        
        if probability < 0.3:
            risk_category = "Low Risk"
        elif probability < 0.6:
            risk_category = "Medium Risk"
        else:
            risk_category = "High Risk"
        
        return PredictionResponse(
            application_id=f"APP_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            default_probability=float(probability),
            prediction="Default" if prediction == 1 else "Non-Default",
            risk_category=risk_category,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch")
def predict_batch(loans: List[LoanApplication]):
    """Make predictions for multiple loan applications"""
    try:
        predictions = []
        
        for loan in loans:
            result = predict(loan)
            predictions.append(result.dict())
        
        return {
            "num_predictions": len(predictions),
            "predictions": predictions,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
