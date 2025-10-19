from locust import HttpUser, task, between
import random
import json


class LoanPredictionUser(HttpUser):
    """
    Load testing user for loan default prediction API
    """
    
    wait_time = between(1, 3)
    
    def on_start(self):
        """Initialize test data"""
        self.sample_data = {
            "Client_Income": random.uniform(20000, 150000),
            "Car_Owned": random.choice([0, 1]),
            "Bike_Owned": random.choice([0, 1]),
            "Active_Loan": random.choice([0, 1]),
            "House_Own": random.choice([0, 1]),
            "Child_Count": random.randint(0, 5),
            "Credit_Amount": random.uniform(5000, 50000),
            "Loan_Annuity": random.uniform(500, 5000),
            "Age_Days": random.uniform(-25000, -7000),
            "Employed_Days": random.uniform(-15000, -100),
            "Client_Income_Type": random.choice(["Working", "Commercial associate", "Pensioner"]),
            "Client_Education": random.choice(["Secondary", "Higher education", "Incomplete higher"]),
            "Client_Marital_Status": random.choice(["M", "S", "D", "W"]),
            "Client_Gender": random.choice(["M", "F"]),
            "Loan_Contract_Type": random.choice(["CL", "RL"]),
            "Client_Housing_Type": random.choice(["House / apartment", "Rented apartment", "With parents"]),
            "Client_Family_Members": random.randint(1, 7),
            "Mobile_Tag": 1,
            "Homephone_Tag": random.choice([0, 1]),
            "Workphone_Working": random.choice([0, 1]),
            "Score_Source_1": random.uniform(0, 1),
            "Score_Source_2": random.uniform(0, 1),
            "Score_Source_3": random.uniform(0, 1),
            "Credit_Bureau": random.randint(0, 10)
        }
    
    @task(3)
    def predict_single(self):
        """Test single prediction endpoint"""
        self.client.post("/predict", json=self.sample_data)
    
    @task(1)
    def health_check(self):
        """Test health check endpoint"""
        self.client.get("/health")
    
    @task(1)
    def model_info(self):
        """Test model info endpoint"""
        self.client.get("/model/info")


if __name__ == "__main__":
    import os
    os.system("locust -f deployment/load_testing.py --host=http://localhost:8000")
