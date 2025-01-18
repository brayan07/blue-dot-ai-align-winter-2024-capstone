from kedro.pipeline import Pipeline, node
from .nodes import filter_raw_quality_data, get_direct_answers, get_answers_with_biases


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
                func=get_direct_answers,
                inputs="quality_filtered_train",
                outputs="quality_filtered_train_with_direct_answers",
                name="get_direct_answers_train"
            ),
            node(
                func=get_answers_with_biases,
                inputs="quality_filtered_train",
                outputs="quality_filtered_train_answers_with_biases",
                name="get_answers_with_biases_train"
            )
        ]
    )