from kedro.pipeline import Pipeline, node
from .nodes import filter_raw_quality_data, partition_data


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=filter_raw_quality_data,
                inputs="params:raw_quality_train_path",
                outputs="quality_filtered_train",
                name="filter_raw_quality_data_train"
            ),
            node(
                func=partition_data,
                inputs="quality_filtered_train",
                outputs="partitioned_quality_filtered_train",
                name="partition_data_train"
            )
        ]
    )