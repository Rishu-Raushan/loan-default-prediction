# Loan Default Prediction - Quick Start Guide

## Setup Instructions

### 1. Create Virtual Environment

Run the setup script:
```powershell
.\setup_environment.bat
```

Or manually:
```powershell
python -m venv venv
.\venv\Scripts\activate.bat
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Verify Installation

```powershell
python -c "import sklearn, pandas, numpy, xgboost, lightgbm, catboost; print('All packages installed successfully')"
```

## Running the Solution

### Option 1: Complete Pipeline (Recommended)

Run the entire pipeline from data loading to model deployment:

```powershell
.\venv\Scripts\activate.bat
python src\main_pipeline.py
```

This will:
- Load and analyze data
- Perform preprocessing
- Engineer features
- Train multiple models
- Evaluate and compare models
- Save best model
- Generate reports and visualizations

### Option 2: Step-by-Step Execution

**Step 1: Exploratory Data Analysis**
```powershell
python notebooks\01_EDA.py
```

**Step 2: Run Main Pipeline**
```powershell
python src\main_pipeline.py
```

## Output Files

After execution, you will find:

### Models
- `models/best_model_*.pkl` - Trained model
- `models/preprocessor.pkl` - Preprocessing pipeline
- `models/feature_engineer.pkl` - Feature engineering pipeline
- `models/feature_names.pkl` - Feature list
- `models/model_metadata.json` - Model information

### Reports
- `reports/model_comparison.csv` - Model performance comparison
- `reports/threshold_optimization.csv` - Threshold analysis
- `reports/business_insights.csv` - Business metrics

### Visualizations
- `reports/plots/target_distribution.png`
- `reports/plots/missing_values.png`
- `reports/plots/correlation_matrix.png`
- `reports/plots/roc_curve_*.png`
- `reports/plots/pr_curve_*.png`
- `reports/plots/confusion_matrix_*.png`
- `reports/plots/feature_importance_*.png`

## Deployment

### Running the API Server

```powershell
.\venv\Scripts\activate.bat
python deployment\app.py
```

Access API at: http://localhost:8000

### API Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `GET /model/info` - Model metadata
- `POST /predict` - Single prediction
- `POST /predict/batch` - Batch predictions

### Testing the API

```powershell
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @sample_request.json
```

### Docker Deployment

```powershell
docker-compose -f deployment\docker-compose.yml up -d
```

Services:
- API: http://localhost:8000
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000

## Running Tests

```powershell
.\venv\Scripts\activate.bat
pytest tests\ -v
```

## Load Testing

```powershell
locust -f deployment\load_testing.py --host=http://localhost:8000
```

Access Locust UI at: http://localhost:8089

## Monitoring

```powershell
python deployment\monitoring.py
```

Metrics available at: http://localhost:9090

## Project Structure

```
Loan Default Prediction/
├── data/                           # Raw data
├── src/                            # Source code
│   ├── data_preprocessing.py       # Data preprocessing
│   ├── feature_engineering.py      # Feature engineering
│   ├── model_training.py           # Model training
│   ├── model_evaluation.py         # Model evaluation
│   └── main_pipeline.py            # Main pipeline
├── notebooks/                      # Analysis notebooks
│   └── 01_EDA.py                   # Exploratory analysis
├── deployment/                     # Deployment files
│   ├── app.py                      # FastAPI application
│   ├── monitoring.py               # Monitoring system
│   ├── load_testing.py             # Load tests
│   ├── Dockerfile                  # Docker configuration
│   └── docker-compose.yml          # Docker Compose
├── tests/                          # Unit tests
├── models/                         # Saved models
├── reports/                        # Reports and plots
├── config/                         # Configuration
├── requirements.txt                # Python dependencies
├── README.md                       # Project overview
├── SOLUTION_APPROACH.md            # Detailed approach
└── SYSTEM_DESIGN.md                # System design
```

## Troubleshooting

### Issue: ModuleNotFoundError

**Solution:** Ensure virtual environment is activated
```powershell
.\venv\Scripts\activate.bat
```

### Issue: Memory Error

**Solution:** Reduce dataset size or increase system memory
```python
# In config/config.yaml
data:
  sample_size: 50000  # Use smaller sample
```

### Issue: Slow Training

**Solution:** Reduce number of models or hyperparameter trials
```yaml
hyperparameter_tuning:
  enable: false  # Disable tuning for faster execution
```

### Issue: Port Already in Use

**Solution:** Change port in deployment
```python
# In deployment/app.py
uvicorn.run(app, host="0.0.0.0", port=8001)
```

## Next Steps

1. Review SOLUTION_APPROACH.md for detailed methodology
2. Review SYSTEM_DESIGN.md for production architecture
3. Customize config/config.yaml for your requirements
4. Deploy to production using Docker
5. Set up monitoring and alerting

## Support

For issues or questions:
1. Check documentation files
2. Review test files for usage examples
3. Examine log files in logs/ directory

## Key Features

- Complete ML pipeline from data to deployment
- Production-ready code with error handling
- Comprehensive testing suite
- Docker containerization
- API for real-time predictions
- Monitoring and alerting
- Load testing framework
- Full documentation

## Performance Expectations

- ROC-AUC: >0.85
- API Latency: <200ms (p95)
- Throughput: 500+ requests/second
- Training Time: 30-60 minutes (full pipeline)
- Prediction Time: <50ms per request

## Best Practices

1. Always activate virtual environment before running
2. Review configuration before training
3. Monitor system resources during training
4. Validate model performance before deployment
5. Test API thoroughly before production use
6. Set up monitoring for production deployment
7. Keep models versioned and documented
8. Regular retraining with new data
