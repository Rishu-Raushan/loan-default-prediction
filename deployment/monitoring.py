import time
import psutil
import numpy as np
from prometheus_client import Counter, Histogram, Gauge, start_http_server
from scipy.stats import ks_2samp
from collections import deque
import joblib


class ModelMonitor:
    """
    Production model monitoring system for drift detection,
    performance tracking, and alerting
    """
    
    def __init__(self, reference_data=None, window_size=1000):
        self.reference_data = reference_data
        self.window_size = window_size
        self.prediction_history = deque(maxlen=window_size)
        self.feature_history = deque(maxlen=window_size)
        
        self.prediction_counter = Counter(
            'predictions_total',
            'Total number of predictions made'
        )
        
        self.prediction_latency = Histogram(
            'prediction_latency_seconds',
            'Prediction latency in seconds'
        )
        
        self.default_rate = Gauge(
            'default_rate',
            'Current default prediction rate'
        )
        
        self.drift_score = Gauge(
            'drift_score',
            'Data drift score (KS statistic)'
        )
        
        self.cpu_usage = Gauge(
            'cpu_usage_percent',
            'CPU usage percentage'
        )
        
        self.memory_usage = Gauge(
            'memory_usage_percent',
            'Memory usage percentage'
        )
    
    def log_prediction(self, features, prediction, probability, latency):
        """Log a prediction for monitoring"""
        self.prediction_counter.inc()
        self.prediction_latency.observe(latency)
        
        self.prediction_history.append({
            'prediction': prediction,
            'probability': probability,
            'timestamp': time.time()
        })
        
        self.feature_history.append(features)
        
        self.update_metrics()
    
    def update_metrics(self):
        """Update monitoring metrics"""
        if len(self.prediction_history) > 0:
            recent_predictions = [p['prediction'] for p in self.prediction_history]
            default_rate_value = np.mean(recent_predictions)
            self.default_rate.set(default_rate_value)
        
        cpu_percent = psutil.cpu_percent(interval=1)
        self.cpu_usage.set(cpu_percent)
        
        memory_percent = psutil.virtual_memory().percent
        self.memory_usage.set(memory_percent)
    
    def detect_data_drift(self, threshold=0.05):
        """Detect data drift using KS test"""
        if self.reference_data is None or len(self.feature_history) < 100:
            return False, 0.0
        
        drift_detected = False
        max_drift_score = 0.0
        
        current_data = np.array(list(self.feature_history))
        
        for i in range(current_data.shape[1]):
            reference_feature = self.reference_data[:, i]
            current_feature = current_data[:, i]
            
            statistic, p_value = ks_2samp(reference_feature, current_feature)
            
            if statistic > max_drift_score:
                max_drift_score = statistic
            
            if p_value < threshold:
                drift_detected = True
                print(f"Drift detected in feature {i}: KS={statistic:.4f}, p-value={p_value:.4f}")
        
        self.drift_score.set(max_drift_score)
        
        return drift_detected, max_drift_score
    
    def detect_performance_degradation(self, threshold=0.1):
        """Detect performance degradation"""
        if len(self.prediction_history) < 100:
            return False
        
        recent_probs = [p['probability'] for p in list(self.prediction_history)[-100:]]
        older_probs = [p['probability'] for p in list(self.prediction_history)[-500:-100]]
        
        if len(older_probs) < 100:
            return False
        
        recent_mean = np.mean(recent_probs)
        older_mean = np.mean(older_probs)
        
        drift = abs(recent_mean - older_mean)
        
        if drift > threshold:
            print(f"Performance degradation detected: drift={drift:.4f}")
            return True
        
        return False
    
    def get_monitoring_report(self):
        """Generate monitoring report"""
        if len(self.prediction_history) == 0:
            return {"status": "No predictions logged yet"}
        
        predictions = [p['prediction'] for p in self.prediction_history]
        probabilities = [p['probability'] for p in self.prediction_history]
        
        report = {
            "total_predictions": len(self.prediction_history),
            "default_rate": np.mean(predictions),
            "avg_probability": np.mean(probabilities),
            "std_probability": np.std(probabilities),
            "cpu_usage": psutil.cpu_percent(),
            "memory_usage": psutil.virtual_memory().percent,
            "timestamp": time.time()
        }
        
        drift_detected, drift_score = self.detect_data_drift()
        report['drift_detected'] = drift_detected
        report['drift_score'] = drift_score
        
        perf_degradation = self.detect_performance_degradation()
        report['performance_degradation'] = perf_degradation
        
        return report


class CanaryDeployment:
    """
    Canary deployment manager for gradual model rollout
    """
    
    def __init__(self, model_v1, model_v2, initial_traffic=0.1):
        self.model_v1 = model_v1
        self.model_v2 = model_v2
        self.traffic_split = initial_traffic
        self.v1_predictions = []
        self.v2_predictions = []
    
    def route_prediction(self, features):
        """Route prediction to appropriate model version"""
        if np.random.random() < self.traffic_split:
            model = self.model_v2
            version = 'v2'
        else:
            model = self.model_v1
            version = 'v1'
        
        start_time = time.time()
        prediction = model.predict(features)
        probability = model.predict_proba(features)[0, 1]
        latency = time.time() - start_time
        
        if version == 'v2':
            self.v2_predictions.append({
                'prediction': prediction[0],
                'probability': probability,
                'latency': latency
            })
        else:
            self.v1_predictions.append({
                'prediction': prediction[0],
                'probability': probability,
                'latency': latency
            })
        
        return prediction, probability, version, latency
    
    def evaluate_canary_performance(self):
        """Evaluate canary model performance vs baseline"""
        if len(self.v1_predictions) < 100 or len(self.v2_predictions) < 100:
            return {"status": "Insufficient data for comparison"}
        
        v1_latencies = [p['latency'] for p in self.v1_predictions]
        v2_latencies = [p['latency'] for p in self.v2_predictions]
        
        v1_probs = [p['probability'] for p in self.v1_predictions]
        v2_probs = [p['probability'] for p in self.v2_predictions]
        
        comparison = {
            'v1_avg_latency': np.mean(v1_latencies),
            'v2_avg_latency': np.mean(v2_latencies),
            'v1_avg_probability': np.mean(v1_probs),
            'v2_avg_probability': np.mean(v2_probs),
            'latency_improvement': (np.mean(v1_latencies) - np.mean(v2_latencies)) / np.mean(v1_latencies) * 100
        }
        
        return comparison
    
    def increase_traffic(self, increment=0.1):
        """Gradually increase traffic to canary model"""
        self.traffic_split = min(1.0, self.traffic_split + increment)
        print(f"Canary traffic increased to {self.traffic_split*100:.1f}%")
    
    def rollback(self):
        """Rollback to previous model version"""
        self.traffic_split = 0.0
        print("Rolled back to model v1")


def start_monitoring_server(port=9090):
    """Start Prometheus metrics server"""
    start_http_server(port)
    print(f"Monitoring server started on port {port}")


if __name__ == "__main__":
    print("Model monitoring and canary deployment system initialized")
    
    start_monitoring_server(port=9090)
    
    monitor = ModelMonitor()
    
    print("Monitoring system ready. Metrics available at http://localhost:9090")
    
    try:
        while True:
            time.sleep(10)
            report = monitor.get_monitoring_report()
            print(f"\nMonitoring Report: {report}")
    except KeyboardInterrupt:
        print("\nMonitoring system stopped")
