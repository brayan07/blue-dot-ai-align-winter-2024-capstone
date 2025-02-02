from collections import defaultdict
from typing import Union, Dict, List

from pydantic import BaseModel

from debate_for_ai_alignment.pipelines.debate.models import (
    UniqueSetNaiveJudgeResults,
    UniqueSetExpertJudgeResults,
    UniqueSetConsultancyResults,
    UniqueSetUnstructuredDebateResults,
    UniqueSetStructuredDebateResults,
)
from debate_for_ai_alignment.pipelines.preprocessing.models import UniqueSet


class ResultsDataManager:
    def __init__(self):
        self.naive_map = {}
        self.expert_map = {}
        self.consultancy_map = {}
        self.unstructured_debate_map = {}
        self.structured_debate_map = {}
        self.unique_set_ids = set()
        self.unique_set_id_to_debate_style_map: Dict[str, List[str]] = defaultdict(list)
        self.unique_set_id_to_metadata_map: Dict[str, UniqueSet] = {}

    def get_available_debate_styles(self, unique_set_id):
        return self.unique_set_id_to_debate_style_map[unique_set_id]

    def get_results(
        self, unique_set_id, debate_style
    ) -> Union[
        UniqueSetNaiveJudgeResults,
        UniqueSetExpertJudgeResults,
        UniqueSetConsultancyResults,
        UniqueSetUnstructuredDebateResults,
        UniqueSetStructuredDebateResults,
    ]:
        if debate_style == "naive":
            return self.naive_map[unique_set_id]
        elif debate_style == "expert":
            return self.expert_map[unique_set_id]
        elif debate_style == "consultancy":
            return self.consultancy_map[unique_set_id]
        elif debate_style == "unstructured_debate":
            return self.unstructured_debate_map[unique_set_id]
        elif debate_style == "structured_debate":
            return self.structured_debate_map[unique_set_id]
        else:
            raise ValueError("Unknown debate style")

    def add_unique_set_metadata(self, input_data: UniqueSet):
        self.unique_set_id_to_metadata_map[input_data.set_unique_id] = input_data

    def add_results(self, results: BaseModel):
        debate_style = None
        if isinstance(results, UniqueSetNaiveJudgeResults):
            self.naive_map[results.results[0].unique_set_id] = results
            debate_style = "naive"
        elif isinstance(results, UniqueSetExpertJudgeResults):
            self.expert_map[results.results[0].unique_set_id] = results
            debate_style = "expert"
        elif isinstance(results, UniqueSetConsultancyResults):
            self.consultancy_map[results.results[0].unique_set_id] = results
            debate_style = "consultancy"
        elif isinstance(results, UniqueSetUnstructuredDebateResults):
            self.unstructured_debate_map[results.results[0].unique_set_id] = results
            debate_style = "unstructured_debate"
        elif isinstance(results, UniqueSetStructuredDebateResults):
            self.structured_debate_map[results.results[0].unique_set_id] = results
            debate_style = "structured_debate"
        else:
            raise ValueError("Unknown results type")
        self.unique_set_ids.add(results.results[0].unique_set_id)
        self.unique_set_id_to_debate_style_map[results.results[0].unique_set_id].append(
            debate_style
        )
