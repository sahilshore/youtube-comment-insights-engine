# app.py

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend before importing pyplot

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import mlflow
import numpy as np
import joblib
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from mlflow.tracking import MlflowClient
import matplotlib.dates as mdates

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"}), 200

# Define the preprocessing function
def preprocess_comment(comment):
    """Apply preprocessing transformations to a comment."""
    try:
        # Convert to lowercase
        comment = comment.lower()

        # Remove trailing and leading whitespaces
        comment = comment.strip()

        # Remove newline characters
        comment = re.sub(r'\n', ' ', comment)

        # Remove non-alphanumeric characters, except punctuation
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

        # Remove stopwords but retain important ones for sentiment analysis
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([word for word in comment.split() if word not in stop_words])
        # Lemmatize the words
        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])

        return comment
    except Exception as e:
        print(f"Error in preprocessing comment: {e}")
        return comment


# Load the model and its ORIGINAL vectorizer from the same MLflow run

def load_model_and_vectorizer(model_name, model_version):

    mlflow.set_tracking_uri(
        "http://ec2-3-111-37-136.ap-south-1.compute.amazonaws.com:5000/"
    )

    client = MlflowClient()

    # Step 1: Get the registered model version information
    version_info = client.get_model_version(
        name=model_name,
        version=str(model_version)
    )

    # Step 2: Retrieve the original training run ID
    run_id = version_info.run_id

    print(f"Loading model version: {model_version}")
    print(f"MLflow training run ID: {run_id}")

    # Step 3: Load the registered model
    model_uri = f"models:/{model_name}/{model_version}"

    model = mlflow.pyfunc.load_model(model_uri)

    # Step 4: Download the EXACT vectorizer saved during this training run
    vectorizer_path = client.download_artifacts(
        run_id,
        "tfidf_vectorizer.pkl"
    )

    # Step 5: Load the vectorizer
    vectorizer = joblib.load(vectorizer_path)

    # Step 6: Verify model and vectorizer feature compatibility
    expected_features = model.metadata.get_input_schema().input_names()

    actual_features = vectorizer.get_feature_names_out().tolist()

    missing_features = set(expected_features) - set(actual_features)
    extra_features = set(actual_features) - set(expected_features)

    print("Model expected features:", len(expected_features))
    print("Vectorizer actual features:", len(actual_features))

    if missing_features or extra_features:
        raise ValueError(
            f"Model/vectorizer mismatch. "
            f"Missing features: {list(missing_features)[:10]}. "
            f"Extra features: {list(extra_features)[:10]}."
        )

    print("Model and vectorizer are compatible!")

    return model, vectorizer


# Initialize the model and vectorizer
model, vectorizer = load_model_and_vectorizer(
    "yt_chrome_plugin_model",
    "13"
)

# Convert comments into the DataFrame format
# expected by the MLflow model.
def predict_comments(comments, batch_size=256):

    all_predictions = []

    # Get the exact feature names from the trained TF-IDF vectorizer
    feature_names = vectorizer.get_feature_names_out()

    for start in range(0, len(comments), batch_size):

        batch_comments = comments[start:start + batch_size]

        # Preprocess comments
        preprocessed_comments = [
            preprocess_comment(comment)
            for comment in batch_comments
        ]

        # Transform comments into a sparse TF-IDF matrix
        transformed_comments = vectorizer.transform(
            preprocessed_comments
        )

        # Validate feature dimensions
        if transformed_comments.shape[1] != len(feature_names):
            raise ValueError(
                f"Feature mismatch: vectorizer produced "
                f"{transformed_comments.shape[1]} features, "
                f"but expected {len(feature_names)}."
            )

        # Convert sparse matrix into a Pandas DataFrame
        # with the exact feature names expected by MLflow
        transformed_df = pd.DataFrame(
            transformed_comments.toarray(),
            columns=feature_names
        )

        # Generate predictions
        batch_predictions = model.predict(transformed_df)

        # Convert predictions into a Python list
        batch_predictions = np.asarray(
            batch_predictions
        ).reshape(-1).tolist()

        all_predictions.extend(batch_predictions)

    return all_predictions

@app.route('/predict_with_timestamps', methods=['POST'])
def predict_with_timestamps():

    data = request.get_json()
    comments_data = data.get('comments')

    if not comments_data:
        return jsonify({"error": "No comments provided"}), 400

    try:

        comments = [
            item['text']
            for item in comments_data
        ]

        timestamps = [
            item['timestamp']
            for item in comments_data
        ]

        # Generate predictions using the same helper function
        predictions = predict_comments(comments)

        # Convert predictions to strings
        predictions = [str(pred) for pred in predictions]

    except Exception as e:

        app.logger.exception("Prediction failed")

        return jsonify({
            "error": f"Prediction failed: {str(e)}"
        }), 500

    response = [
        {
            "comment": comment,
            "sentiment": sentiment,
            "timestamp": timestamp
        }
        for comment, sentiment, timestamp in zip(
            comments,
            predictions,
            timestamps
        )
    ]

    return jsonify(response)

@app.route('/predict', methods=['POST'])
def predict():

    data = request.get_json()
    comments = data.get('comments')

    if not comments:
        return jsonify({"error": "No comments provided"}), 400

    try:

        # Generate predictions using the helper function
        predictions = predict_comments(comments)

        # Convert predictions to strings
        predictions = [str(pred) for pred in predictions]

    except Exception as e:

        app.logger.exception("Prediction failed")

        return jsonify({
            "error": f"Prediction failed: {str(e)}"
        }), 500

    response = [
        {
            "comment": comment,
            "sentiment": sentiment
        }
        for comment, sentiment in zip(comments, predictions)
    ]

    return jsonify(response)

@app.route('/generate_chart', methods=['POST'])
def generate_chart():
    try:
        data = request.get_json()
        sentiment_counts = data.get('sentiment_counts')
        
        if not sentiment_counts:
            return jsonify({"error": "No sentiment counts provided"}), 400

        # Prepare data for the pie chart
        labels = ['Positive', 'Neutral', 'Negative']
        sizes = [
            int(sentiment_counts.get('1', 0)),
            int(sentiment_counts.get('0', 0)),
            int(sentiment_counts.get('-1', 0))
        ]
        if sum(sizes) == 0:
            raise ValueError("Sentiment counts sum to zero")
        
        colors = ['#36A2EB', '#C9CBCF', '#FF6384']  # Blue, Gray, Red

        # Generate the pie chart
        plt.figure(figsize=(6, 6))
        plt.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=140,
            textprops={'color': 'w'}
        )
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        # Save the chart to a BytesIO object
        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG', transparent=True)
        img_io.seek(0)
        plt.close()

        # Return the image as a response
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_chart: {e}")
        return jsonify({"error": f"Chart generation failed: {str(e)}"}), 500

@app.route('/generate_wordcloud', methods=['POST'])
def generate_wordcloud():
    try:
        data = request.get_json()
        comments = data.get('comments')

        if not comments:
            return jsonify({"error": "No comments provided"}), 400

        # Preprocess comments
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]

        # Combine all comments into a single string
        text = ' '.join(preprocessed_comments)

        # Generate the word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='black',
            colormap='Blues',
            stopwords=set(stopwords.words('english')),
            collocations=False
        ).generate(text)

        # Save the word cloud to a BytesIO object
        img_io = io.BytesIO()
        wordcloud.to_image().save(img_io, format='PNG')
        img_io.seek(0)

        # Return the image as a response
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_wordcloud: {e}")
        return jsonify({"error": f"Word cloud generation failed: {str(e)}"}), 500

@app.route('/generate_trend_graph', methods=['POST'])
def generate_trend_graph():
    try:
        data = request.get_json()
        sentiment_data = data.get('sentiment_data')

        if not sentiment_data:
            return jsonify({"error": "No sentiment data provided"}), 400

        # Convert sentiment_data to DataFrame
        df = pd.DataFrame(sentiment_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Set the timestamp as the index
        df.set_index('timestamp', inplace=True)

        # Ensure the 'sentiment' column is numeric
        df['sentiment'] = df['sentiment'].astype(int)

        # Map sentiment values to labels
        sentiment_labels = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}

        # Resample the data over monthly intervals and count sentiments
        monthly_counts = df.resample('M')['sentiment'].value_counts().unstack(fill_value=0)

        # Calculate total counts per month
        monthly_totals = monthly_counts.sum(axis=1)

        # Calculate percentages
        monthly_percentages = (monthly_counts.T / monthly_totals).T * 100

        # Ensure all sentiment columns are present
        for sentiment_value in [-1, 0, 1]:
            if sentiment_value not in monthly_percentages.columns:
                monthly_percentages[sentiment_value] = 0

        # Sort columns by sentiment value
        monthly_percentages = monthly_percentages[[-1, 0, 1]]

        # Plotting
        plt.figure(figsize=(12, 6))

        colors = {
            -1: 'red',     # Negative sentiment
            0: 'gray',     # Neutral sentiment
            1: 'green'     # Positive sentiment
        }

        for sentiment_value in [-1, 0, 1]:
            plt.plot(
                monthly_percentages.index,
                monthly_percentages[sentiment_value],
                marker='o',
                linestyle='-',
                label=sentiment_labels[sentiment_value],
                color=colors[sentiment_value]
            )

        plt.title('Monthly Sentiment Percentage Over Time')
        plt.xlabel('Month')
        plt.ylabel('Percentage of Comments (%)')
        plt.grid(True)
        plt.xticks(rotation=45)

        # Format the x-axis dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))

        plt.legend()
        plt.tight_layout()

        # Save the trend graph to a BytesIO object
        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG')
        img_io.seek(0)
        plt.close()

        # Return the image as a response
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_trend_graph: {e}")
        return jsonify({"error": f"Trend graph generation failed: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)