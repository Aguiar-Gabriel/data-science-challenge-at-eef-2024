from kedro.pipeline import Pipeline, node, pipeline
from .nodes import preprocess_public_dataframe, preprocess_public_dataframe_sem_pca

def create_pipeline_sem_pca(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=preprocess_public_dataframe_sem_pca,
            inputs="public",
            outputs=[
                "X_train_rf",
                "y_train_rf",
                "X_to_predict_rf",
                "flightid_pred",
            ],
            name="preprocess_public_dataframe_node",
        ),
    ])

def create_pipeline_com_pca(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=preprocess_public_dataframe,
            inputs=["public", "features_pca"],
            outputs=[
                "X_train_rf",
                "y_train_rf",
                "X_to_predict_rf",
                "flightid_pred",
            ],
            name="preprocess_public_dataframe_node",
        ),
    ])