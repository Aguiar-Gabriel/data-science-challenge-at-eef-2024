from kedro.pipeline import Pipeline, node, pipeline
from .nodes import load_public_dataset, preprocess_public_dataframe

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=load_public_dataset,
            inputs="public",
            outputs="public_loaded",
            name="load_public_node"
        ),
        node(
            func=preprocess_public_dataframe,
            inputs="public_loaded",
            outputs=[
                "X_train_rf",
                "y_train_rf",
                "X_to_predict_rf",
                "flightid_pred",
            ],
            name="preprocess_public_dataframe_node",
        ),
    ])
