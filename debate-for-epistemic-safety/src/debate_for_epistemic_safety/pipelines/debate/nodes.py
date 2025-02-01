import asyncio
from logging import getLogger
from typing import Dict, Callable, Union

from debate_for_epistemic_safety.pipelines.debate.consultancy import run_consultancy
from debate_for_epistemic_safety.pipelines.debate.expert import run_expert
from debate_for_epistemic_safety.pipelines.debate.models import QuestionDebateResults, ArticleDebateResults, \
    UniqueSetNaiveJudgeResults, UniqueSetExpertJudgeResults, UniqueSetConsultancyResults, \
    UniqueSetUnstructuredDebateResults, UniqueSetStructuredDebateResults
from debate_for_epistemic_safety.pipelines.debate.naive import run_naive
from debate_for_epistemic_safety.pipelines.debate.unstructured_debate import run_unstructured_debate
from debate_for_epistemic_safety.pipelines.debate.structured_debate import run_structured_debate
from debate_for_epistemic_safety.pipelines.preprocessing.models import UniqueSet


def run_naive_for_unique_set(unique_set: Dict, llm_config: Dict) -> Dict:
    unique_set = UniqueSet(**unique_set)
    return asyncio.run(_run_naive_for_unique_set(unique_set, llm_config)).model_dump()

def run_expert_for_unique_set(unique_set: Dict, llm_config: Dict) -> Dict:
    unique_set = UniqueSet(**unique_set)
    return asyncio.run(_run_expert_for_unique_set(unique_set, llm_config)).model_dump()

def run_consultancy_for_unique_set(unique_set: Dict, llm_config: Dict) -> Dict:
    unique_set = UniqueSet(**unique_set)
    return asyncio.run(_run_consultancy_for_unique_set(unique_set, llm_config)).model_dump()

def run_unstructured_debate_for_unique_set(unique_set: Dict, llm_config: Dict) -> Dict:
    unique_set = UniqueSet(**unique_set)
    return asyncio.run(_run_unstructured_debate_for_unique_set(unique_set, llm_config)).model_dump()

def run_structured_debate_for_unique_set(unique_set: Dict, llm_config: Dict) -> Dict:
    unique_set = UniqueSet(**unique_set)
    return asyncio.run(_run_structured_debate_for_unique_set(unique_set, llm_config)).model_dump()

async def _run_expert_for_unique_set(unique_set: UniqueSet, llm_config: Dict) -> UniqueSetExpertJudgeResults:
    expert_result_futures = []
    for question_idx in range(len(unique_set.questions)):
        for is_correct_option_first in [True, False]:
            expert_result_futures.append(run_expert(unique_set, question_idx, is_correct_option_first, llm_config))
    # Wait for all the results to be ready and then create a UniqueSetExpertJudgeResults
    expert_results = await asyncio.gather(*expert_result_futures)
    return UniqueSetExpertJudgeResults(
        unique_set_id=unique_set.article_id,
        results=expert_results
    )

async def _run_naive_for_unique_set(unique_set: UniqueSet, llm_config: Dict) -> UniqueSetNaiveJudgeResults:
    naive_result_futures = []
    for question_idx in range(len(unique_set.questions)):
        for is_correct_option_first in [True, False]:
            naive_result_futures.append(run_naive(unique_set, question_idx, is_correct_option_first, llm_config))
    # Wait for all the results to be ready and then create a UniqueSetNaiveJudgeResults
    naive_results = await asyncio.gather(*naive_result_futures)
    return UniqueSetNaiveJudgeResults(
        unique_set_id=unique_set.article_id,
        results=naive_results
    )

async def _run_consultancy_for_unique_set(unique_set: UniqueSet, llm_config: Dict) -> UniqueSetConsultancyResults:
    consultancy_result_futures = []
    for question_idx in range(len(unique_set.questions)):
        for is_correct_option_first in [True, False]:
            for is_defending_correct in [True, False]:
                for n_rounds in [4]:
                    consultancy_result_futures.append(
                        run_consultancy(
                            unique_set,
                            question_idx,
                            is_correct_option_first,
                            is_defending_correct,
                            n_rounds,
                            llm_config
                        )
                    )
    # Wait for all the results to be ready and then create a UniqueSetConsultancyResults
    consultancy_results = await asyncio.gather(*consultancy_result_futures)
    return UniqueSetConsultancyResults(
        unique_set_id=unique_set.article_id,
        results=consultancy_results
    )

async def _run_unstructured_debate_for_unique_set(unique_set: UniqueSet, llm_config: Dict) -> UniqueSetUnstructuredDebateResults:
    unstructured_debate_result_futures = []
    for question_idx in range(len(unique_set.questions)):
        for is_correct_option_first in [True, False]:
            for n_rounds in [5]:
                for is_agent_defending_correct_option_first in [True, False]:
                    unstructured_debate_result_futures.append(
                        run_unstructured_debate(
                            unique_set,
                            question_idx,
                            is_correct_option_first,
                            is_agent_defending_correct_option_first,
                            n_rounds,
                            llm_config
                        )
                    )
    # Wait for all the results to be ready and then create a UniqueSetUnstructuredDebateResults
    unstructured_debate_results = await asyncio.gather(*unstructured_debate_result_futures)
    return UniqueSetUnstructuredDebateResults(
        unique_set_id=unique_set.article_id,
        results=unstructured_debate_results
    )

async def _run_structured_debate_for_unique_set(unique_set: UniqueSet, llm_config: Dict) -> UniqueSetStructuredDebateResults:
    structured_debate_result_futures = []
    for question_idx in range(len(unique_set.questions)):
        for is_correct_option_first in [True, False]:
            for n_rounds in [5]:
                for is_agent_defending_correct_option_first in [True, False]:
                    structured_debate_result_futures.append(
                        run_structured_debate(
                            unique_set,
                            question_idx,
                            is_correct_option_first,
                            is_agent_defending_correct_option_first,
                            n_rounds,
                            llm_config
                        )
                    )
    # Wait for all the results to be ready and then create a UniqueSetUnstructuredDebateResults
    structured_debate_results = await asyncio.gather(*structured_debate_result_futures)
    return UniqueSetStructuredDebateResults(
        unique_set_id=unique_set.article_id,
        results=structured_debate_results
    )
