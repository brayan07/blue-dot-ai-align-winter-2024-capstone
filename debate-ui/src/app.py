import logging
import pathlib
from typing import List

import dash
import numpy as np
from autogen_core.models import TopLogprob
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc


from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project

from debate_for_ai_alignment.pipelines.debate.models import (
    UniqueSetNaiveJudgeResults,
    UniqueSetExpertJudgeResults,
    UniqueSetConsultancyResults,
    UniqueSetUnstructuredDebateResults,
    UniqueSetStructuredDebateResults,
    DebateResult,
)
from debate_for_ai_alignment.pipelines.preprocessing.models import (
    UniqueSet,
)
from .data_manager import ResultsDataManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NAME_TO_RESULT_CLASS_MAP = {
    "naive": UniqueSetNaiveJudgeResults,
    "expert": UniqueSetExpertJudgeResults,
    "consultancy": UniqueSetConsultancyResults,
    "unstructured_debate": UniqueSetUnstructuredDebateResults,
    "structured_debate": UniqueSetStructuredDebateResults,
}


def _normalize_log_prob_to_probability(
    log_prob: List[TopLogprob], is_correct_option_first: bool
) -> float:
    correct_answer = "A" if is_correct_option_first else "B"
    incorrect_answer = "B" if is_correct_option_first else "A"
    token_to_log_prob = {}
    for lp in log_prob:
        token = bytes(lp.bytes).decode("utf-8")
        token_to_log_prob[token] = lp.logprob
    correct_log_prob = token_to_log_prob[correct_answer]
    incorrect_log_prob = token_to_log_prob[incorrect_answer]
    return np.exp(correct_log_prob) / (
        np.exp(correct_log_prob) + np.exp(incorrect_log_prob)
    )


APP_INSTRUCTIONS = (
    "## Introduction\n"
    "This application allows you to review the results of different debate protocols on a set of questions "
    "from the [QuALITY dataset](https://github.com/nyu-mll/quality).\n"
    "## Instructions\n"
    "1. Select a unique set ID from the dropdown menu. These ids corresponds to a unique set of questions written by the same author on a single article."
    "  Different 'unique-sets' may contain questions on the same article.\n"
    "2. Select a question from the dropdown menu. Unique sets may contain 1-3 questions. This will display the question text, the correct answer, and the best distractor in the `Article Information` section.\n"
    "3. Select a debate style from the dropdown menu, whether the correct option is listed first as 'A' or second as 'B', and any other options in the dropdown menus that appear. This will display the debate messages and the confidence of the agents in the `Debate` section.\n"
    "4. Click the `Collapse/Expand Article Content` button to hide or show the full article content. This is the text that is shown to the debaters and consultants, but not the non-expert judges.\n"
    "5. Click the `Collapse/Expand Debate` button to hide or show the debate messages and confidence scores.\n"
    "## Confidence Scores\n"
    "The confidence scores are calculated by taking the log probability of the correct answer and the incorrect answer from the language model, and normalizing them to a probability between 0 and 1.\n"
    "We sample the log probabilities of the correct and incorrect answers from the language model at the end of each round of debate to get a sense of how these change over the course of the debate.\n"
)


class ResultsViewingApplication:
    def __init__(self, app: dash.Dash, project_path: str):
        self.project_path = project_path
        self.app = app
        self.data_manager = ResultsDataManager()
        self._load_data()
        self._set_app_layout()
        self._add_callbacks()

    def _load_data_kind(self, kind: str, catalog):
        data = catalog.load(f"partitioned_unique_set_{kind}_results@incremental")
        for raw_result in data.values():
            results = NAME_TO_RESULT_CLASS_MAP[kind](**raw_result)
            self.data_manager.add_results(results)

    def _load_metadata(self, catalog):
        data = catalog.load("partitioned_quality_filtered_train")
        for load_unique_set_metadata_func in data.values():
            raw_metadata = load_unique_set_metadata_func()
            metadata = UniqueSet(**raw_metadata)
            self.data_manager.add_unique_set_metadata(metadata)

    def _load_data(self):
        bootstrap_project(self.project_path)
        with KedroSession.create(project_path=self.project_path) as session:
            context = session.load_context()
            catalog = context.catalog
            self._load_metadata(catalog)
            self._load_data_kind("naive", catalog)
            self._load_data_kind("expert", catalog)
            self._load_data_kind("consultancy", catalog)
            self._load_data_kind("unstructured_debate", catalog)
            self._load_data_kind("structured_debate", catalog)

    def _add_callbacks(self):
        @self.app.callback(
            Output("question-dropdown", "options"),
            Input("unique-set-dropdown", "value"),
        )
        def update_question_options(unique_set_id):
            if unique_set_id is None:
                return []
            return self._get_question_options(unique_set_id)

        @self.app.callback(
            Output("debate-style-dropdown", "options"),
            Input("unique-set-dropdown", "value"),
        )
        def update_debate_style_options(unique_set_id):
            if unique_set_id is None:
                return []
            return self._get_debate_style_options(unique_set_id)

        @self.app.callback(
            Output("article-info", "children"),
            Input("unique-set-dropdown", "value"),
            Input("question-dropdown", "value"),
        )
        def update_article_information(unique_set_id, question_idx):
            if unique_set_id is None or question_idx is None:
                return ""
            return self._get_article_information_text(unique_set_id, question_idx)

        @self.app.callback(
            Output("debate-container", "children"),
            Output("toggle-debate-button", "style"),
            Input("unique-set-dropdown", "value"),
            Input("question-dropdown", "value"),
            Input("debate-style-dropdown", "value"),
            Input("correct-option-order-dropdown", "value"),
            Input("is-agent-defending-correct-option-first-dropdown", "value"),
            Input("is-consultant-defending-correct-option", "value"),
        )
        def update_debate_container(
            unique_set_id,
            question_idx,
            debate_style,
            correct_option_order,
            is_agent_defending_correct_option_first,
            is_consultant_defending_correct_option,
        ):
            if (
                unique_set_id is None
                or question_idx is None
                or debate_style is None
                or correct_option_order is None
            ):
                return [], {"display": "none"}
            if (
                debate_style == "consultancy"
                and is_consultant_defending_correct_option is None
            ):
                return [], {"display": "none"}
            if (
                debate_style in ["unstructured_debate", "structured_debate"]
                and is_agent_defending_correct_option_first is None
            ):
                return [], {"display": "none"}
            return self._update_debate_container(
                unique_set_id,
                question_idx,
                debate_style,
                correct_option_order,
                is_agent_defending_correct_option_first,
                is_consultant_defending_correct_option,
            ), {}

        @self.app.callback(
            Output("correct-option-order-dropdown", "options"),
            Input("unique-set-dropdown", "value"),
            Input("question-dropdown", "value"),
            Input("debate-style-dropdown", "value"),
        )
        def update_correct_option_order_options(
            unique_set_id, question_idx, debate_style
        ):
            if unique_set_id is None or question_idx is None or debate_style is None:
                return []
            return self._get_correct_option_order_options(
                unique_set_id, question_idx, debate_style
            )

        @self.app.callback(
            Output("is-agent-defending-correct-option-first-dropdown", "options"),
            Input("unique-set-dropdown", "value"),
            Input("question-dropdown", "value"),
            Input("debate-style-dropdown", "value"),
            Input("correct-option-order-dropdown", "value"),
        )
        def update_order_of_agents_options(
            unique_set_id, question_idx, debate_style, correct_option_order
        ):
            if (
                unique_set_id is None
                or question_idx is None
                or debate_style is None
                or correct_option_order is None
            ):
                return []
            return self._get_order_of_agents_options(
                unique_set_id, question_idx, debate_style, correct_option_order
            )

        @self.app.callback(
            Output(
                "is-agent-defending-correct-option-first-dropdown-container", "style"
            ),
            Input("debate-style-dropdown", "value"),
        )
        def update_appearance_of_agent_order_dropdown(debate_style):
            if debate_style not in ["unstructured_debate", "structured_debate"]:
                return {"display": "none"}
            return {}

        @self.app.callback(
            Output(
                "is-consultant-defending-correct-option-first-dropdown-container",
                "style",
            ),
            Input("debate-style-dropdown", "value"),
        )
        def update_appearance_of_consultant_order_dropdown(debate_style):
            if debate_style not in ["consultancy"]:
                return {"display": "none"}
            return {}

        @self.app.callback(
            Output("is-consultant-defending-correct-option", "options"),
            Input("unique-set-dropdown", "value"),
            Input("question-dropdown", "value"),
            Input("debate-style-dropdown", "value"),
            Input("correct-option-order-dropdown", "value"),
        )
        def update_consultant_defending_correct_option_first_dropdown(
            unique_set_id, question_idx, debate_style, correct_option_order
        ):
            if (
                unique_set_id is None
                or question_idx is None
                or debate_style is None
                or correct_option_order is None
            ):
                return []
            return self._get_consultant_defending_correct_option_first_options(
                unique_set_id, question_idx, debate_style
            )

        @self.app.callback(
            Output("debate-collapse", "is_open"),
            Input("toggle-debate-button", "n_clicks"),
            State("debate-collapse", "is_open"),
        )
        def toggle_debate_container(n_clicks, is_open):
            if n_clicks:
                return not is_open
            return is_open

        @self.app.callback(
            Output("article-collapse", "is_open"),
            Input("toggle-article-content-button", "n_clicks"),
            State("article-collapse", "is_open"),
        )
        def toggle_article_content(n_clicks, is_open):
            if n_clicks:
                return not is_open
            return is_open

        @self.app.callback(
            Output("article-content", "children"),
            Input("unique-set-dropdown", "value"),
        )
        def update_article_content(unique_set_id):
            if unique_set_id is None:
                return ""
            metadata: UniqueSet = self.data_manager.unique_set_id_to_metadata_map[
                unique_set_id
            ]
            return metadata.article

    def _get_consultant_defending_correct_option_first_options(
        self, unique_set_id, question_idx, debate_style
    ):
        if debate_style not in ["consultancy"]:
            return []
        results = self.data_manager.get_results(unique_set_id, debate_style).results
        possible_options = set()
        for result in results:
            if result.question_idx == question_idx:
                possible_options.add(result.is_defending_correct)
        return [
            {
                "label": f"Consultant defending {'correct' if option else 'incorrect'} option",
                "value": option,
            }
            for option in possible_options
        ]

    def _get_order_of_agents_options(
        self, unique_set_id, question_idx, debate_style, correct_option_order
    ):
        if debate_style not in ["unstructured_debate", "structured_debate"]:
            return []
        results = self.data_manager.get_results(unique_set_id, debate_style).results
        possible_options = set()
        for result in results:
            if (
                result.question_idx == question_idx
                and result.is_correct_option_first == correct_option_order
            ):
                possible_options.add(result.is_agent_defending_correct_option_first)
        return [
            {
                "label": f"Agent defending correct option {'first' if option else 'second'}",
                "value": option,
            }
            for option in possible_options
        ]

    def _get_debate_style_options(self, unique_set_id):
        return [
            {"label": f"{debate_style}", "value": debate_style}
            for debate_style in self.data_manager.get_available_debate_styles(
                unique_set_id
            )
        ]

    def _get_correct_option_order_options(
        self, unique_set_id, question_idx, debate_style
    ):
        results = self.data_manager.get_results(unique_set_id, debate_style).results
        possible_options = set()
        for result in results:
            if result.question_idx == question_idx:
                possible_options.add(result.is_correct_option_first)
        return [
            {
                "label": f"Correct option {'first' if option else 'second'}",
                "value": option,
            }
            for option in possible_options
        ]

    def _get_article_information_text(self, unique_set_id, question_idx):
        metadata: UniqueSet = self.data_manager.unique_set_id_to_metadata_map[
            unique_set_id
        ]
        question_text = metadata.questions[question_idx].question
        correct_option_idx = metadata.questions[question_idx].gold_label - 1
        best_distractor_idx = (
            metadata.questions[question_idx].validation[0].untimed_best_distractor - 1
        )
        correct_answer_text = metadata.questions[question_idx].options[
            correct_option_idx
        ]
        distractor_answer_text = metadata.questions[question_idx].options[
            best_distractor_idx
        ]
        return (
            f"Article Title: {metadata.title}\n"
            f"Question: {question_text}\n"
            f"Correct Answer: {correct_answer_text}\n"
            f"Best Distractor: {distractor_answer_text}"
        )

    def _get_question_options(self, unique_set_id):
        questions = self.data_manager.unique_set_id_to_metadata_map[
            unique_set_id
        ].questions
        return [{"label": f"Question {i}", "value": i} for i in range(len(questions))]

    def _update_debate_container(
        self,
        unique_set_id,
        question_idx,
        debate_style,
        correct_option_order,
        is_agent_defending_correct_option_first,
        is_consultant_defending_correct_option,
    ):
        result = self._get_result_matching_criteria(
            unique_set_id,
            question_idx,
            correct_option_order,
            debate_style,
            is_agent_defending_correct_option_first,
            is_consultant_defending_correct_option,
        )
        if debate_style == "consultancy":
            return self._format_consultancy_messages(result)
        elif debate_style in ["unstructured_debate", "structured_debate"]:
            return self._format_debate_messages(result)
        elif debate_style in ["naive", "expert"]:
            return self._format_single_agent_result(result)
        else:
            raise ValueError("Unknown debate style")

    def _format_single_agent_result(self, result: DebateResult):
        return dbc.Row(
            [
                dbc.Col(
                    [
                        html.H3("Round 1"),
                        self._get_confidence_card(round_n=0, result=result),
                    ]
                )
            ]
        )

    def _get_result_matching_criteria(
        self,
        unique_set_id,
        question_idx,
        correct_option_order,
        debate_style,
        is_agent_defending_correct_option_first,
        is_consultant_defending_correct_option,
    ):
        results = self.data_manager.get_results(unique_set_id, debate_style).results
        results = [
            result
            for result in results
            if result.question_idx == question_idx
            and result.is_correct_option_first == correct_option_order
        ]
        if debate_style == "consultancy":
            results = [
                result
                for result in results
                if result.is_defending_correct == is_consultant_defending_correct_option
            ]
        elif debate_style in ["unstructured_debate", "structured_debate"]:
            results = [
                result
                for result in results
                if result.is_agent_defending_correct_option_first
                == is_agent_defending_correct_option_first
            ]
        result = results[0]
        return result

    def _format_debate_messages(self, result: DebateResult):
        # In debate each round is 3 messages, but the first message is an instruction
        messages = result.task_result.messages[1:]
        components = []
        total_rounds = len(messages) // 3
        for i in range(total_rounds):
            cards = [
                dbc.Card(
                    [
                        dbc.CardHeader(messages[i * 3 + j].source),
                        dbc.CardBody(
                            # Don't change text editing
                            dcc.Markdown(
                                messages[i * 3 + j].content,
                                style={"white-space": "pre-wrap"},
                            )
                        ),
                    ]
                )
                for j in range(3)
            ]
            confidence_card = self._get_confidence_card(round_n=i, result=result)
            components.append(
                dbc.Row(
                    dbc.Card(
                        [
                            dbc.Col(
                                [
                                    html.H3(f"Round {i+1}"),
                                    dbc.Row(
                                        [
                                            dbc.Col(cards[0]),
                                            dbc.Col(cards[1]),
                                            dbc.Col(cards[2]),
                                            dbc.Col(confidence_card),
                                        ]
                                    ),
                                ]
                            )
                        ]
                    )
                )
            )
        return components

    def _get_confidence_card(self, round_n: int, result: DebateResult):
        round_log_prob = result.logprob[round_n]
        correct_option_probability = _normalize_log_prob_to_probability(
            round_log_prob, is_correct_option_first=result.is_correct_option_first
        )
        confidence_card = dbc.Card(
            [
                dbc.CardHeader("Confidence"),
                dbc.CardBody(
                    [
                        dbc.Row(
                            [
                                dbc.Col([html.Pre("Correct Option Probability: ")]),
                                dbc.Col(
                                    [
                                        # Label probability as percent and make it green
                                        dbc.Progress(
                                            value=correct_option_probability,
                                            min=0,
                                            max=1,
                                            label=f"{correct_option_probability:.2%}",
                                            color="success",
                                        ),
                                    ]
                                ),
                            ]
                        ),
                        # Add the same for incorrect option
                        dbc.Row(
                            [
                                dbc.Col([html.Pre("Incorrect Option Probability: ")]),
                                dbc.Col(
                                    [
                                        dbc.Progress(
                                            value=1 - correct_option_probability,
                                            min=0,
                                            max=1,
                                            label=f"{1 - correct_option_probability:.2%}",
                                            color="danger",
                                        ),
                                    ]
                                ),
                            ]
                        ),
                    ]
                ),
            ]
        )
        return confidence_card

    def _format_consultancy_messages(self, result: DebateResult):
        # 1 round is 2 messages, but first message is omitted
        messages = result.task_result.messages[1:]
        components = []
        total_rounds = len(messages) // 2
        for i in range(total_rounds):
            cards = [
                dbc.Card(
                    [
                        dbc.CardHeader(messages[i * 2 + j].source),
                        dbc.CardBody(
                            dcc.Markdown(
                                messages[i * 2 + j].content,
                                style={"white-space": "pre-wrap"},
                            )
                        ),
                    ]
                )
                for j in range(2)
            ]
            confidence_card = self._get_confidence_card(round_n=i, result=result)
            components.append(
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H3(f"Round {i+1}"),
                                dbc.Row(
                                    [
                                        dbc.Col(cards[0]),
                                        dbc.Col(cards[1]),
                                        dbc.Col(confidence_card),
                                    ]
                                ),
                            ]
                        )
                    ]
                )
            )
        return components

    def _set_app_layout(self):
        self.app.layout = dbc.Container(
            [
                dbc.Row([html.H1("Review of Debate Results")]),
                dbc.Row([dbc.Card(dcc.Markdown(APP_INSTRUCTIONS))]),
                dbc.Row(
                    [
                        dbc.Card(
                            [
                                dbc.Col(
                                    html.Div(
                                        [
                                            html.H2("Unique Set ID"),
                                            dcc.Dropdown(
                                                id="unique-set-dropdown",
                                                options=[
                                                    {"label": i, "value": i}
                                                    for i in sorted(
                                                        self.data_manager.unique_set_ids
                                                    )
                                                ],
                                                placeholder="Select a unique set ID.",
                                            ),
                                        ]
                                    ),
                                ),
                                dbc.Col(
                                    html.Div(
                                        [
                                            html.H2("Question"),
                                            dcc.Dropdown(
                                                id="question-dropdown",
                                                options=[],
                                                placeholder="Select a question",
                                            ),
                                        ]
                                    ),
                                ),
                            ]
                        )
                    ]
                ),
                html.Br(),
                dbc.Row([html.H2("Article Information")]),
                dbc.Row(
                    dbc.Card(
                        [
                            dbc.Col(
                                [
                                    html.H3("Article & Question Metadata"),
                                    dbc.Card(
                                        [
                                            # text box, making sure if there are multiple lines, it will be scrollable
                                            html.Pre(id="article-info")
                                        ]
                                    ),
                                    html.H3("Article Content"),
                                    # Collapse button for article content
                                    dbc.Button(
                                        "Collapse/Expand Article Content",
                                        id="toggle-article-content-button",
                                        color="primary",
                                        className="mb-3",
                                    ),
                                    dbc.Collapse(
                                        [
                                            dbc.Card(
                                                [
                                                    # Use markdown to render the article content, but make the box a fixed size and scrollable
                                                    dcc.Markdown(
                                                        id="article-content",
                                                        style={
                                                            "height": "500px",
                                                            "overflow": "scroll",
                                                        },
                                                    )
                                                ]
                                            ),
                                        ],
                                        id="article-collapse",
                                        is_open=False,
                                    ),
                                ]
                            ),
                        ]
                    )
                ),
                html.Br(),
                dbc.Row(
                    [
                        html.H2("Protocol Settings"),
                    ]
                ),
                html.Br(),
                dbc.Row(
                    [
                        dbc.Card(
                            [
                                # Debate Style picker
                                dbc.Col(
                                    [
                                        html.H3("Debate Style"),
                                        dcc.Dropdown(
                                            id="debate-style-dropdown",
                                            placeholder="Select a debate style",
                                        ),
                                    ]
                                ),
                                dbc.Col(
                                    [
                                        # Selector for whether correct option is presented first
                                        html.H3("Correct Option Order"),
                                        dcc.Dropdown(
                                            id="correct-option-order-dropdown",
                                            options=[
                                                # {"label": "Correct option first", "value": True},
                                                # {"label": "Incorrect option first", "value": False},
                                            ],
                                            placeholder="Select the order of the correct option",
                                        ),
                                    ]
                                ),
                                dbc.Col(
                                    [
                                        # Drop down asking ofor value of is_agent_defending_correct_option_first
                                        html.H3("Order of Agents"),
                                        dcc.Dropdown(
                                            id="is-agent-defending-correct-option-first-dropdown",
                                            options=[
                                                {
                                                    "label": "Agent defending correct option goes first",
                                                    "value": True,
                                                },
                                                {
                                                    "label": "Agent is defending incorrect option first",
                                                    "value": False,
                                                },
                                            ],
                                            placeholder="Select which agent goes first",
                                        ),
                                    ],
                                    id="is-agent-defending-correct-option-first-dropdown-container",
                                ),
                                dbc.Col(
                                    [
                                        # Drop down asking ofor value of is_agent_defending_correct_option_first
                                        html.H3("Consultant's Position"),
                                        dcc.Dropdown(
                                            id="is-consultant-defending-correct-option",
                                            options=[],
                                            placeholder="Select which agent goes first",
                                        ),
                                    ],
                                    id="is-consultant-defending-correct-option-first-dropdown-container",
                                ),
                            ]
                        )
                    ]
                ),
                html.Br(),
                dbc.Row([html.H2("Debate")]),
                dbc.Row(
                    [
                        dbc.Button(
                            "Collapse/Expand Debate",
                            id="toggle-debate-button",
                            color="primary",
                            className="mb-3",
                        ),
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Collapse(
                            [
                                dbc.Card(
                                    id="debate-container",
                                    children=[],
                                )
                            ],
                            id="debate-collapse",
                            is_open=True,
                        )
                    ]
                ),
            ]
        )


def _initialize_results_viewer():
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    root_dir = pathlib.Path(__file__).parent.parent.parent.resolve()
    return ResultsViewingApplication(
        app, (root_dir / "debate-for-ai-alignment").as_posix()
    )


results_viewer = _initialize_results_viewer()
server = results_viewer.app.server

# Run the app in development mode
if __name__ == "__main__":
    results_viewer.app.run(host="0.0.0.0", port=8050, debug=True)
