from typing import List

from autogen_core.models import ChatCompletionTokenLogprob


def get_context_from_article(article, question_idx, is_correct_option_first):
    article_title = article.title
    question_text = article.questions[question_idx].question
    correct_option_idx = article.questions[question_idx].gold_label - 1
    best_distractor_idx = (
        article.questions[question_idx].validation[0].untimed_best_distractor - 1
    )
    correct_answer_text = article.questions[question_idx].options[correct_option_idx]
    distractor_answer_text = article.questions[question_idx].options[
        best_distractor_idx
    ]
    if is_correct_option_first:
        first_option = correct_answer_text
        second_option = distractor_answer_text
    else:
        first_option = distractor_answer_text
        second_option = correct_answer_text
    return article_title, question_text, first_option, second_option


def _get_top_log_probs_for_answer(log_probs: List[ChatCompletionTokenLogprob]):
    # Get log prob values for the answer token, which should be either A or B
    # Seek the ':' token.
    colon_idx = None
    for i, log_prob in enumerate(log_probs):
        if ":" in log_prob.token.strip():
            colon_idx = i
            break
    if colon_idx is None:
        raise ValueError("Could not find ':' token in log probs")
    # Get the log prob values for the answer token, which should be the next token that is an A or B
    # Seek the A or B token
    answer_idx = None
    for i, log_prob in enumerate(log_probs[colon_idx + 1 :]):
        if log_prob.token.strip() in ["A", "B"]:
            answer_idx = i
            break
    if answer_idx is None:
        raise ValueError("Could not find A or B token in log probs")
    # Get the top log prob values for the answer token
    return log_probs[colon_idx + 1 + answer_idx].top_logprobs
