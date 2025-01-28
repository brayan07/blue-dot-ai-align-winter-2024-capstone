from .evaluator import Evaluator
import logging

class EvaluatorConsultantBaseline(Evaluator):
    def __init__(self, article, llm_config, debug=False):
        super().__init__(article, llm_config, debug)

    @property
    def logger(self):
        return logging.getLogger(__name__)


    def generate_argument(self, question_idx: int, answer_idx: int):
        # TODO: Implement using similar technique to that used by the debate paper.
        pass