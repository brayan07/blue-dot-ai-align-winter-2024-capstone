import inspect
from typing import Any, Literal, List

import autogen_agentchat.messages as agentchat_messages
from autogen_agentchat.base import TaskResult
from autogen_agentchat.messages import BaseMessage
from autogen_core.models import TopLogprob
from pydantic import BaseModel, root_validator, model_validator, field_validator, Field

from debate_for_ai_alignment.pipelines.preprocessing.models import UniqueSet

# Build a mapping of type name -> concrete message class for deserialization.
_MESSAGE_TYPE_MAP = {
    name: cls
    for name, cls in inspect.getmembers(agentchat_messages, inspect.isclass)
    if issubclass(cls, BaseMessage) and not inspect.isabstract(cls)
}


class DebateResult(BaseModel):
    unique_set_id: str
    question_idx: int
    answer: List[Literal["A", "B"]]
    logprob: List[List[TopLogprob]]


class NaiveJudgeResults(DebateResult):
    is_correct_option_first: bool


class ExpertJudgeResults(DebateResult):
    is_correct_option_first: bool


class MultipleRoundDebateResult(DebateResult):
    is_correct_option_first: bool
    n_rounds: int
    task_result: TaskResult

    @field_validator("task_result", mode="before")
    @classmethod
    def deserialize_task_result(cls, v: Any) -> Any:
        """Handle deserialization of TaskResult from dict.

        Pydantic cannot deserialize the abstract BaseAgentEvent/BaseChatMessage
        union directly. This validator maps each message dict to its concrete
        type using the 'type' discriminator field before passing to TaskResult.
        """
        if isinstance(v, dict) and "messages" in v:
            messages = []
            for msg in v["messages"]:
                if isinstance(msg, dict) and "type" in msg:
                    msg_cls = _MESSAGE_TYPE_MAP.get(msg["type"])
                    if msg_cls is not None:
                        messages.append(msg_cls.model_validate(msg))
                        continue
                messages.append(msg)
            v = {**v, "messages": messages}
        return v

    @model_validator(mode="after")
    def check_lengths(self):
        if len(self.answer) != self.n_rounds:
            raise ValueError(
                f"Expected {self.n_rounds} rounds, got {len(self.answer)} answers."
            )
        if len(self.logprob) != self.n_rounds:
            raise ValueError(
                f"Expected {self.n_rounds} rounds, got {len(self.logprob)} logprobs."
            )
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
    answer: Literal["A", "B"]


class ResponseModelWithLogProb(ResponseModel):
    top_log_probs: List[TopLogprob]


class QuestionForAgent(BaseModel):
    selected_speaker: Literal["debate_agent_1", "debate_agent_2", "both"] = Field(
        description="The agent to address the question to. If 'both', the question will be asked to both agents."
    )
    question: str = Field(
        description="The question to be asked of the selected speaker. Does not include addresses to the agent as these are added automatically as part of scaffolding."
    )
