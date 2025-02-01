from typing import Literal, List

from autogen_agentchat.base import TaskResult
from autogen_core.models import TopLogprob
from pydantic import BaseModel, root_validator, model_validator

from debate_for_ai_alignment.pipelines.preprocessing.models import UniqueSet


class DebateResult(BaseModel):
    unique_set_id: str
    question_idx: int
    answer: List[Literal['A', 'B']]
    logprob: List[List[TopLogprob]]


class NaiveJudgeResults(DebateResult):
    is_correct_option_first: bool

class ExpertJudgeResults(DebateResult):
    is_correct_option_first: bool

class MultipleRoundDebateResult(DebateResult):
    is_correct_option_first: bool
    n_rounds: int
    task_result: TaskResult

    @model_validator(mode='after')
    def check_lengths(self):
        if len(self.answer) != self.n_rounds:
            raise ValueError(f"Expected {self.n_rounds} rounds, got {len(self.answer)} answers.")
        if len(self.logprob) != self.n_rounds:
            raise ValueError(f"Expected {self.n_rounds} rounds, got {len(self.logprob)} logprobs.")
        return self

class ConsultancyResults(MultipleRoundDebateResult):
    is_defending_correct: bool

class UnstructuredDebateResults(MultipleRoundDebateResult):
    is_agent_defending_correct_option_first: bool

class StructuredDebateResults(MultipleRoundDebateResult):
    is_agent_defending_correct_option_first: bool

class UniqueSetNaiveJudgeResults(BaseModel):
    unique_set_id: str
    results: List[NaiveJudgeResults]

class UniqueSetExpertJudgeResults(BaseModel):
    unique_set_id: str
    results: List[ExpertJudgeResults]

class UniqueSetConsultancyResults(BaseModel):
    unique_set_id: str
    results: List[ConsultancyResults]

class UniqueSetUnstructuredDebateResults(BaseModel):
    unique_set_id: str
    results: List[UnstructuredDebateResults]

class UniqueSetStructuredDebateResults(BaseModel):
    unique_set_id: str
    results: List[StructuredDebateResults]

class QuestionDebateResults(BaseModel):
    question_idx: int
    naive_judge: List[NaiveJudgeResults]
    expert_judge: List[ExpertJudgeResults]
    consultancy: List[ConsultancyResults]
    unstructured_debate: List[UnstructuredDebateResults]
    structured_debate: List[StructuredDebateResults]


class ArticleDebateResults(BaseModel):
    article: UniqueSet
    results: List[QuestionDebateResults]


class ResponseModel(BaseModel):
    answer: Literal['A', 'B']


class ResponseModelWithLogProb(ResponseModel):
    top_log_probs: List[TopLogprob]
