import mlflow
import joblib
from mlflow.tracking import MlflowClient

# Configuration
MODEL_NAME = "yt_chrome_plugin_model"
MODEL_VERSION = "13"
VECTOR_PATH = "./tfidf_vectorizer.pkl"

# MLflow configuration
mlflow.set_tracking_uri(
    "http://ec2-3-111-37-136.ap-south-1.compute.amazonaws.com:5000/"
)

client = MlflowClient()

print("\n========== LOADING MODEL ==========")

model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
model = mlflow.pyfunc.load_model(model_uri)

print("Model loaded successfully!")

print("\n========== LOADING VECTORIZER ==========")

vectorizer = joblib.load(VECTOR_PATH)

print("Vectorizer loaded successfully!")

# Get expected model input features
schema = model.metadata.get_input_schema()
expected_features = schema.input_names()

# Get actual vectorizer features
actual_features = vectorizer.get_feature_names_out().tolist()

print("\n========== FEATURE COMPARISON ==========")

print("Model expected features:", len(expected_features))
print("Vectorizer actual features:", len(actual_features))

missing_features = sorted(
    set(expected_features) - set(actual_features)
)

extra_features = sorted(
    set(actual_features) - set(expected_features)
)

print("\nMissing features:", len(missing_features))
print(missing_features[:30])

print("\nExtra features:", len(extra_features))
print(extra_features[:30])

print("\nFeature order identical:", expected_features == actual_features)

# Find the MLflow run associated with model version 13
print("\n========== MODEL VERSION INFORMATION ==========")

model_version_info = client.get_model_version(
    name=MODEL_NAME,
    version=MODEL_VERSION
)

run_id = model_version_info.run_id

print("Model version:", model_version_info.version)
print("Model run ID:", run_id)

# List artifacts associated with the model's training run
print("\n========== TRAINING RUN ARTIFACTS ==========")

def show_artifacts(path=""):
    artifacts = client.list_artifacts(run_id, path)

    for artifact in artifacts:
        print(artifact.path)

        if artifact.is_dir:
            show_artifacts(artifact.path)

show_artifacts()

print("\n========== DIAGNOSTIC COMPLETE ==========")