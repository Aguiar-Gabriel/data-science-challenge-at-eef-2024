import logging
from kedro.pipeline import Pipeline, node, pipeline
from kedro.config import OmegaConfigLoader
from kedro.framework.project import settings
from kedro.io import DataCatalog, MemoryDataset
from kedro_datasets import pickle, pandas

from .nodes import train_model, tune_and_train_model, predict_espera, build_submission

def create_pipeline(**kwargs) -> Pipeline:
    """Cria a pipeline de data science dinamicamente."""
    logger = logging.getLogger(__name__)

    conf_loader = OmegaConfigLoader(conf_source=str(settings.CONF_SOURCE))
    parameters = conf_loader["parameters"]
    
    models_to_train = parameters.get("models", {})
    tuning_params = parameters.get("hyperparameter_tuning", {})
    tuning_enabled = tuning_params.get("enabled", False)
    models_to_tune = tuning_params.get("models_to_tune", [])

    if not models_to_train:
        logger.warning("Nenhum modelo definido em 'parameters.models'. Pipeline de data science estará vazio.")
        return pipeline([])

    model_pipelines = []
    for model_name, model_config in models_to_train.items():
        # Decide se vai otimizar ou treinar diretamente
        if tuning_enabled and model_name in models_to_tune:
            train_node_func = tune_and_train_model
            train_node_inputs = ["X_train_rf", "y_train_rf", f"params:models.{model_name}", "params:hyperparameter_tuning"]
        else:
            train_node_func = train_model
            train_node_inputs = ["X_train_rf", "y_train_rf", f"params:models.{model_name}"]

        model_dataset_name = f"{model_name}_model"
        predictions_dataset_name = f"{model_name}_predictions"

        model_pipeline = pipeline(
            [
                node(
                    func=train_node_func,
                    inputs=train_node_inputs,
                    outputs=model_dataset_name,
                    name=f"train_{model_name}_node",
                ),
                node(
                    func=predict_espera,
                    inputs=[model_dataset_name, "X_to_predict_rf"],
                    outputs=predictions_dataset_name,
                    name=f"predict_{model_name}_node",
                ),
            ]
        )
        model_pipelines.append(model_pipeline)

    # Adiciona um nó para criar o arquivo de submissão final com o melhor modelo (catboost)
    submission_pipeline = pipeline([
        node(
            func=build_submission,
            inputs=["flightid_pred", "catboost_predictions"],
            outputs="submission",
            name="build_submission_node",
        )
    ])

    return sum(model_pipelines, Pipeline([])) + submission_pipeline