from kedro.pipeline import Pipeline, node, pipeline

from .nodes import train_random_forest, predict_espera, build_submission


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=train_random_forest,
                inputs=["X_train_rf", "y_train_rf", "params:model_rf"],
                outputs="rf_model",
                name="train_random_forest_node",
            ),
            node(
                func=predict_espera,
                inputs=["rf_model", "X_to_predict_rf"],
                outputs="espera_pred",
                name="predict_espera_node",
            ),
            node(
                func=build_submission,
                inputs=["flightid_pred", "espera_pred"],
                outputs="submission",
                name="build_submission_node",
            ),
        ]
    )
