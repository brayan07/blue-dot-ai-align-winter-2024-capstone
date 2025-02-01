import json
from typing import Any, Dict

from debate_for_ai_alignment.pipelines.preprocessing.models import (
    UniqueSet,
    QualityData,
)


def filter_raw_quality_data(path: str):
    quality_data = _load_data_from_path(path)
    filtered_sets = []
    for unique_set in quality_data.unique_sets:
        # 0. Only sourced from the Gutenberg dataset
        if unique_set.source != "Gutenberg":
            continue
        filtered_questions = []
        for question in unique_set.questions:
            average_context_required = sum(
                val.untimed_eval2_context for val in question.validation
            ) / len(question.validation)
            best_distractors = [
                val.untimed_best_distractor for val in question.validation
            ]
            all_best_distractors_the_same = len(set(best_distractors)) == 1
            if (
                # 1. 100% of untimed annotators chose the correct answer
                all(
                    val.untimed_answer == question.gold_label
                    for val in question.validation
                )
                # 2. Less than 50% of timed annotators chose the correct answer
                and question.difficult == True
                # 3. All untimed annotators agree that the question is answerable and unambiguous
                and all(
                    val.untimed_eval1_answerability == 1 for val in question.validation
                )
                # 4. Average ”context required” rating from untimed annotators is at least 1.5
                and average_context_required >= 1.5
                # 5. Writer label matches the gold label (the answer voted as the correct answer by annotators matches what the question writer labelled as the correct answer)
                and question.gold_label == question.writer_label
                # 6. All best_distractors are the same and are not the correct answer
                and all_best_distractors_the_same
                and best_distractors[0] != question.gold_label
            ):
                filtered_questions.append(question)
        if filtered_questions:
            filtered_sets.append(
                UniqueSet(
                    article_id=unique_set.article_id,
                    set_unique_id=unique_set.set_unique_id,
                    source=unique_set.source,
                    title=unique_set.title,
                    author=unique_set.author,
                    article=unique_set.article,
                    questions=filtered_questions,
                )
            )
    return QualityData(unique_sets=filtered_sets).model_dump()


def partition_data(data: Dict) -> Dict[str, Dict[str, Any]]:
    """Partition data by unique_set_id"""
    data = QualityData(**data)
    partitions = {}
    for unique_set in data.unique_sets:
        partitions[
            f"{unique_set.set_unique_id}/unique_set.json"
        ] = unique_set.model_dump()
    return partitions


def _load_data_from_path(path: str) -> QualityData:
    json_str = "["
    with open(path, "r") as file:
        row_strs = [line for line in file]
    json_str += ",".join(row_strs) + "]"
    return QualityData(unique_sets=json.loads(json_str))
