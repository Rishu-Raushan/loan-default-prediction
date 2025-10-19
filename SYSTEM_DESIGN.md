# System Design Document - Loan Default Prediction

## 1. Production Architecture Design

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Load Balancer (nginx)                     │
└────────────────────────┬────────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
    ┌────▼───┐      ┌────▼───┐     ┌────▼───┐
    │  API   │      │  API   │     │  API   │
    │ Server │      │ Server │     │ Server │
    │   v1   │      │   v2   │     │   v2   │
    └────┬───┘      └────┬───┘     └────┬───┘
         │               │               │
         └───────────────┼───────────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
    ┌────▼───────┐  ┌────▼─────────┐  ┌─▼──────────┐
    │   Model    │  │  Monitoring  │  │  Database  │
    │  Registry  │  │   (Prom +    │  │  (Postgres)│
    │  (MLflow)  │  │   Grafana)   │  │            │
    └────────────┘  └──────────────┘  └────────────┘
```

### Components

**1. API Gateway Layer**
- nginx load balancer
- SSL termination
- Rate limiting
- Request routing

**2. Application Layer**
- FastAPI microservices
- Horizontal scaling with multiple workers
- Health check endpoints
- Async request handling

**3. Model Serving Layer**
- Model versioning with MLflow
- Model artifact storage
- A/B testing capability
- Rollback mechanism

**4. Monitoring Layer**
- Prometheus for metrics collection
- Grafana for visualization
- Custom metrics (latency, throughput, drift)
- Alert manager for notifications

**5. Data Layer**
- PostgreSQL for prediction logging
- Redis for caching
- S3/MinIO for model artifacts

## 2. Canary Deployment Strategy

### Implementation Steps

**Phase 1: Preparation (Week 1)**
1. Deploy new model version (v2) alongside existing model (v1)
2. Route 10% of traffic to v2
3. Monitor key metrics for 48 hours
4. Compare performance with v1

**Phase 2: Gradual Rollout (Week 2)**
1. If metrics are stable, increase to 25% traffic
2. Monitor for 24 hours
3. Increase to 50% traffic
4. Monitor for 24 hours

**Phase 3: Full Deployment (Week 3)**
1. If all metrics are satisfactory, route 100% to v2
2. Keep v1 running for 1 week as fallback
3. Implement automated rollback if issues detected

### Rollback Criteria
- Response latency increase > 20%
- Error rate increase > 5%
- Data drift score > 0.1
- Prediction distribution shift > 15%
- User-reported issues > threshold

### Code Implementation

```python
class CanaryDeployment:
    def __init__(self, model_v1, model_v2, initial_traffic=0.1):
        self.model_v1 = model_v1
        self.model_v2 = model_v2
        self.traffic_split = initial_traffic
    
    def route_prediction(self, features):
        if random.random() < self.traffic_split:
            return self.model_v2.predict(features), 'v2'
        else:
            return self.model_v1.predict(features), 'v1'
    
    def evaluate_and_promote(self):
        # Compare metrics between versions
        # Automatic promotion if v2 outperforms v1
        pass
```

## 3. ML Model Monitoring Strategy

### Monitoring Components

**1. Performance Monitoring**
- Prediction latency (p50, p95, p99)
- Throughput (requests/second)
- Error rates
- Model accuracy metrics

**2. Data Drift Detection**
- Feature distribution comparison
- KS test for numerical features
- Chi-square test for categorical features
- PSI (Population Stability Index)

**3. Model Drift Detection**
- Prediction distribution over time
- Confusion matrix tracking
- ROC-AUC monitoring
- Precision/Recall trends

**4. Business Metrics**
- Default rate predictions
- False positive/negative costs
- Revenue impact
- Risk-adjusted returns

### Alert Rules

```yaml
alerts:
  - name: HighLatency
    condition: p95_latency > 500ms
    action: notify_team
  
  - name: DataDrift
    condition: ks_statistic > 0.1
    action: trigger_retraining
  
  - name: PerformanceDegradation
    condition: roc_auc_drop > 5%
    action: rollback_model
  
  - name: HighErrorRate
    condition: error_rate > 5%
    action: immediate_alert
```

### Monitoring Implementation

```python
class ModelMonitor:
    def detect_data_drift(self, reference_data, current_data):
        for feature in features:
            statistic, p_value = ks_2samp(
                reference_data[feature],
                current_data[feature]
            )
            if p_value < 0.05:
                alert("Data drift detected", feature)
    
    def track_performance(self, predictions, actuals):
        roc_auc = roc_auc_score(actuals, predictions)
        self.roc_auc_gauge.set(roc_auc)
        
        if roc_auc < self.threshold:
            trigger_retraining()
```

## 4. Load and Stress Testing

### Testing Strategy

**1. Load Testing**
- Normal traffic: 100 requests/second
- Peak traffic: 500 requests/second
- Sustained load: 8 hours
- Tool: Locust

**2. Stress Testing**
- Gradually increase load until system breaks
- Identify bottlenecks
- Determine maximum capacity
- Recovery testing

**3. Spike Testing**
- Sudden traffic increase (10x normal)
- System response time
- Auto-scaling verification

### Testing Scenarios

```python
# Locust load test configuration
class LoanPredictionUser(HttpUser):
    wait_time = between(1, 3)
    
    @task
    def predict(self):
        self.client.post("/predict", json=loan_data)

# Run with: locust --users 1000 --spawn-rate 10
```

### Performance Benchmarks

| Metric | Target | Maximum Acceptable |
|--------|--------|-------------------|
| Response Time (p95) | < 200ms | < 500ms |
| Throughput | 500 req/s | 1000 req/s |
| Error Rate | < 0.1% | < 1% |
| CPU Usage | < 70% | < 85% |
| Memory Usage | < 80% | < 90% |

## 5. ML Training Tracking

### Experiment Tracking with MLflow

**1. Track Everything**
- Hyperparameters
- Metrics (train/validation/test)
- Model artifacts
- Data versions
- Feature importance
- Training duration

**2. MLflow Implementation**

```python
import mlflow

with mlflow.start_run():
    # Log parameters
    mlflow.log_params({
        'n_estimators': 200,
        'max_depth': 7,
        'learning_rate': 0.05
    })
    
    # Train model
    model.fit(X_train, y_train)
    
    # Log metrics
    mlflow.log_metrics({
        'train_roc_auc': train_score,
        'test_roc_auc': test_score,
        'precision': precision,
        'recall': recall
    })
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
    
    # Log artifacts
    mlflow.log_artifact("feature_importance.png")
```

**3. Model Registry**
- Version control for models
- Stage transitions (Staging → Production)
- Model lineage tracking
- A/B testing support

### Audit Trail

**Requirements**
- Who trained the model
- When it was trained
- What data was used
- What hyperparameters were used
- Performance metrics
- Approval workflow

**Implementation**

```python
training_metadata = {
    'timestamp': datetime.now(),
    'user': get_current_user(),
    'data_version': '2024-01-15',
    'git_commit': get_git_commit(),
    'environment': get_environment_info(),
    'metrics': {
        'roc_auc': 0.87,
        'precision': 0.82,
        'recall': 0.79
    }
}

log_to_database(training_metadata)
```

## 6. Continuous Delivery and Automation

### CI/CD Pipeline

```yaml
# GitHub Actions / GitLab CI
stages:
  - test
  - train
  - evaluate
  - deploy

test:
  - run unit tests
  - run integration tests
  - check code quality
  - validate data schema

train:
  - trigger training on new data
  - log experiments to MLflow
  - run cross-validation
  - generate reports

evaluate:
  - compare with baseline model
  - check performance thresholds
  - validate on holdout set
  - generate evaluation report

deploy:
  - package model
  - update model registry
  - canary deployment
  - smoke tests
```

### Automated Retraining

**Triggers**
1. Performance degradation detected
2. Data drift exceeds threshold
3. Scheduled (monthly)
4. New data availability

**Retraining Workflow**

```python
def automated_retraining_pipeline():
    # 1. Check triggers
    if should_retrain():
        # 2. Fetch new data
        new_data = fetch_latest_data()
        
        # 3. Validate data
        validate_data_quality(new_data)
        
        # 4. Train new model
        new_model = train_model(new_data)
        
        # 5. Evaluate
        if new_model.score > current_model.score:
            # 6. Deploy via canary
            deploy_canary(new_model)
        else:
            log_warning("New model underperforms")
```

### Infrastructure as Code

```python
# Terraform for infrastructure
resource "aws_ecs_service" "model_api" {
  name = "loan-prediction-api"
  cluster = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.api.arn
  desired_count = 3
  
  load_balancer {
    target_group_arn = aws_lb_target_group.api.arn
    container_name = "api"
    container_port = 8000
  }
}
```

## 7. Security and Compliance

### Security Measures
- API authentication (JWT tokens)
- Rate limiting
- Input validation
- SQL injection prevention
- Encryption at rest and in transit
- GDPR compliance for data handling

### Compliance Tracking
- Model fairness audits
- Bias detection
- Explainability reports (SHAP values)
- Regulatory approval workflow

## 8. Disaster Recovery

### Backup Strategy
- Daily model artifacts backup
- Database replication
- Multi-region deployment
- Automated failover

### Recovery Procedures
- RPO (Recovery Point Objective): 1 hour
- RTO (Recovery Time Objective): 15 minutes
- Documented runbooks
- Regular DR drills

## Summary

This system design provides a robust, scalable, and maintainable production ML system with:
- High availability through load balancing
- Safe deployments via canary releases
- Comprehensive monitoring and alerting
- Automated testing and deployment
- Full audit trail and compliance
- Disaster recovery capabilities

The architecture supports continuous improvement while maintaining system stability and reliability.
