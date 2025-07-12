from kedro.pipeline import Pipeline, node, pipeline
from .nodes import preprocess_public_dataframe, load_and_split_data

def create_pipeline_sem_pca(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=load_and_split_data,
            inputs="csv_data",
            outputs=[
                "X_train_rf",
                "y_train_rf",
                "X_to_predict_rf",
                "flightid_pred",
            ],
            name="load_and_split_data_node",
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