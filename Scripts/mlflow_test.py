import mlflow
import random

mlflow.set_tracking_uri("http://ec2-3-111-37-136.ap-south-1.compute.amazonaws.com:5000/")

with mlflow.start_run():
    mlflow.log_param("param1", random.randint(1, 100))
    mlflow.log_param("param2", random.random())
    mlflow.log_metric("metric1", random.random())
    mlflow.log_metric("metric2", random.uniform(0, 1))


    print("Logged parameters and metrics to MLflow.")