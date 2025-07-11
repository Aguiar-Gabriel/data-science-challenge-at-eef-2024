from spaceflights.pipelines import data_processing, data_science

def register_pipelines():
    dp_sem_pca = data_processing.create_pipeline_sem_pca()
    dp_com_pca = data_processing.create_pipeline_com_pca()
    ds_pipeline = data_science.create_pipeline()

    return {
        "dp_sem_pca": dp_sem_pca,
        "dp_com_pca": dp_com_pca,
        "data_science": ds_pipeline,
        "__default__": dp_sem_pca + ds_pipeline,
    }