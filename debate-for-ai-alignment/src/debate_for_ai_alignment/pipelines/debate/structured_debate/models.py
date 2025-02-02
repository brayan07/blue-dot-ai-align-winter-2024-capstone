from typing import Annotated, Optional, List

from pydantic import BaseModel, Field, AfterValidator


def ensure_in_quotes(value: str) -> str:
    if (
        value
        and not (value.startswith('"') and value.endswith('"'))
        and not (value.startswith("“") and value.endswith("”"))
    ):
        value = value.strip()
        value = f"“{value}”"
    return value


class SupportingFact(BaseModel):
    fact: str = Field(
        description="A fact supporting the claim. This should be a plain factual statement about the article or events therein. Do not include opinions or interpretations about the fact."
    )
    quote: Annotated[Optional[str], AfterValidator(ensure_in_quotes)] = Field(
        description="A verbatim quote from the article that supports the fact. This is optional, and should be included only if advantageous to the argument."
    )


class Premise(BaseModel):
    premise: str = Field(
        description="A premise that supports the conclusion. This should be a logical statement that, along with the other premises, leads to the conclusion."
    )
    supporting_facts: List[SupportingFact] = Field(
        description="Supporting facts that back up the premise. These are not required if the premise is self-evident. Limited to a maximum of 2 supporting facts.",
    )


class StructuredArgument(BaseModel):
    premises: List[Premise] = Field(
        description="A list of premises that support the conclusion. These should be logical statements that, taken together, should clearly support the conclusion, even for someone who hasn't read the article. Limited to a maximum of 3 premises.",
    )
    conclusion: str = Field(
        description="The conclusion that the premises lead to. This should be a clear statement that follows logically from the premises."
    )


class CounterPremise(BaseModel):
    counter_premise: str = Field(
        description="Statement expressing what is wrong with the opponent's premise."
    )
    supporting_facts: List[SupportingFact] = Field(
        description="Supporting facts that back up the counter-premise. These are not required if the counter-premise is self-evident. Limited to a maximum of 2 supporting facts.",
    )


class StructuredRebuttal(BaseModel):
    counter_premises: List[CounterPremise] = Field(
        description="A list of counter-premises that refute the opponent's premises. These should be logical statements that, taken together, should clearly refute the opponent's argument. Limited to a maximum of 3 counter-premises.",
    )
    conclusion: str = Field(
        description="The conclusion that the counter-premises lead to. This should be a clear statement that follows logically from the counter-premises."
    )


class ResponseToQuestion(BaseModel):
    answer: str = Field(
        description="The answer to the question. This should be a clear statement that directly answers the question."
    )
    supporting_facts: List[SupportingFact] = Field(
        description="Supporting facts that back up the answer. These are not required if the answer is self-evident. Limited to a maximum of 2 supporting facts.",
    )
