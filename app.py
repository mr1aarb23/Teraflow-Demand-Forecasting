from flask import Flask, request, jsonify
from src.pipeline.prediction import ModelForecast, ModelForecastConfig
import os
import pandas as pd
import json

app = Flask(__name__)


@app.route('/train', methods=['POST'])
def train():
    try:
        if not request.is_json:
            app.logger.error("Request data is not in JSON format.")
            return "Request data is not in JSON format.", 400

        data = request.get_json()
        df = pd.DataFrame(data)
        df.to_csv('src/data.csv', index=False)

        app.logger.info("Received JSON Data")
        os.system("python main.py")
        return "The Forecast Model is Ready!"
    except Exception as e:
        app.logger.error(f"An error occurred: {e}")
        return f"An error occurred: {e}", 500


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if not request.is_json:
            app.logger.error("Request data is not in JSON format.")
            return "Request data is not in JSON format.", 400

        data = request.get_json()
        df = pd.DataFrame(data)
        df.to_csv('src/Predict_data.csv', index=False)
        config = ModelForecastConfig(df)
        prediction_pipeline = ModelForecast(config)
        forecast_batch_rounded, product_ids = prediction_pipeline.predict()
        forecast_df = prediction_pipeline.create_forecast_table(
            forecast_batch_rounded, product_ids)
        app.logger.info(forecast_df)

        predictions = []
        for i, product_id in enumerate(product_ids):
            predictions.append({
                "Product id": str(product_id),
                "Quantity": forecast_batch_rounded[i].tolist()
            })

        return jsonify(predictions)
    except Exception as e:
        app.logger.error(f"An error occurred: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
