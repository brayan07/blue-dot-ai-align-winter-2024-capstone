from typing import List

from pydantic import BaseModel


class Validation(BaseModel):
    untimed_answer: int
    untimed_eval1_answerability: int
    untimed_eval2_context: int
    untimed_best_distractor: int

class Question(BaseModel):
    question: str
    options: List[str]
    gold_label: int
    writer_label: int
    validation: List[Validation]
    difficult: bool


class UniqueSet(BaseModel):
    article_id: str
    set_unique_id: str
    source: str
    title: str
    author: str
    article: str
    questions: List[Question]


class QualityData(BaseModel):
    unique_sets: List[UniqueSet]


class LLMConfig(BaseModel):
    model: str
    api_key: str
    temperature: float = 0.2
