from src.utils.common import load_json
import os
from pathlib import Path
from src.entity.config_entity import ModelSelectionConfig
import pandas as pd
import shutil


class ModelSelection:
    def __init__(self, config: ModelSelectionConfig):
        self.config = config

    def load_evalation_metrics_files(self):
        models_evaluation_metrics = []
        evaluation_metrics = {}

        for file in os.listdir(self.config.eval_dir):
            eval_file = load_json(
                Path(os.path.join(self.config.eval_dir, file)))
            models_evaluation_metrics.append(eval_file)

        for i, config_box in enumerate(models_evaluation_metrics):
            evaluation_metrics[config_box.Model] = {
                'MSE': config_box.Mean_Val_MSE,
                'RMSE': config_box.Mean_Val_RMSE,
                'MAE': config_box.Mean_Val_MAE,
                'SMAPE': config_box.Mean_Val_SMAPE,
                'Runtime': config_box.Mean_Val_Training_time
            }
        return evaluation_metrics

    def get_evaluation_score(self, evaluation_metrics):
        weights = {'MSE': 0.15, 'RMSE': 0.35,
                   'MAE': 0.15, 'SMAPE': 0.2, 'Runtime': 0.15}
        metrics_df = pd.DataFrame(evaluation_metrics).T

        normalized_metrics = (metrics_df - metrics_df.min()) / \
            (metrics_df.max() - metrics_df.min())

        weighted_metrics = normalized_metrics * pd.Series(weights)

        weighted_metrics['Total_Score'] = weighted_metrics.sum(axis=1)

        ranked_metrics = weighted_metrics.sort_values(by='Total_Score')

        best_model = ranked_metrics.index[0]
        forecast_model_path = os.path.join(self.config.models_dir, best_model)
        forecast_model = str(os.listdir(forecast_model_path)[0])

        shutil.copy(os.path.join(forecast_model_path, forecast_model),
                    os.path.join(self.config.root_dir, "Forecast_model"))
        ranked_metrics.to_csv(os.path.join(
            self.config.root_dir, 'ranked_metrics.csv'))
