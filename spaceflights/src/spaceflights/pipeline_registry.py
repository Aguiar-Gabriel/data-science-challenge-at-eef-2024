from spaceflights.pipelines import data_processing, data_science

def register_pipelines():
    return {
        "data_processing": data_processing.create_pipeline(),
        "data_science": data_science.create_pipeline(),
        "__default__": data_processing.create_pipeline() + data_science.create_pipeline(),
    }
