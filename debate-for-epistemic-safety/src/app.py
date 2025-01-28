import logging

import dash
import pandas as pd
from dash import dcc, html, Input, Output, State, ALL
import dash_bootstrap_components as dbc
import json
import os

from debate_for_epistemic_safety.pipelines.master.evaluator import Evaluator, LLMConfig
from debate_for_epistemic_safety.pipelines.master.nodes import QualityData

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Load your articles and questions data
with open('data/01_raw/quality_filtered_train.json') as f:
    data = json.load(f)
    quality_data = QualityData(**data)

from pathlib import Path

from kedro.config import OmegaConfigLoader, MissingConfigException
from kedro.framework.project import settings

PROJECT_ROOT = "/Users/bjaramillo/PycharmProjects/blue-dot-ai-align-winter-2024-capstone/debate-for-epistemic-safety"
conf_path = str(Path(PROJECT_ROOT) / settings.CONF_SOURCE)
conf_loader = OmegaConfigLoader(conf_source=conf_path)

try:
    credentials = conf_loader["credentials"]
except MissingConfigException:
    credentials = {}

# Extract article titles for the dropdown
article_titles = [article.title for article in quality_data.unique_sets]
llm_config = LLMConfig(
    model="gpt-4o-mini",
    api_key=credentials["open_ai_api"]["key"]
)


def generate_option_x_content(letter:str):
    content = [
        dbc.Row(
            [
                dbc.Col([
                    html.H2(f"Option {letter.capitalize()}"),
                    html.P(id=f'option-{letter}-text')
                    ],
                ),
            ],
        ),
        dbc.Row([
            dbc.Col([
                dcc.Loading(
                    id=f'loading-{letter}',
                    type='default',
                    children=[
                        dcc.Textarea(
                            id=f'option-{letter}-unstructured-argument',
                            placeholder="Enter your argument or generate one...",
                        ),
                        html.Br(),
                        dbc.Button("Generate Argument", id=f'generate-argument-{letter}-button', color='primary'),
                    ]
                ),
            ]),
            dbc.Col([
                dcc.Loading([
                    dbc.Container([html.Div(id=f'evaluation-table-{letter}')]),
                    html.Br(),
                    dbc.Button("Review Evidence", id=f'review-evidence-{letter}-button', color='primary'),
                ]),
            ]),
            dbc.Col([
                dcc.Textarea(
                    id=f'option-{letter}-evidence-review',
                    placeholder="Generate an argument and click on one of the evidence links to review evidence.",
                )
            ])
        ]),
    ]
    return content

# Define the layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Argument Evaluator Demo"),
            dcc.Dropdown(
                id='article-dropdown',
                options=[{'label': title, 'value': title} for title in article_titles],
                placeholder="Select an article"
            ),
            dcc.Dropdown(
                id='question-dropdown',
                placeholder="Select a question"
            ),
        ])
    ]),
    html.Br(),
    *generate_option_x_content('a'),
    html.Br(),
    *generate_option_x_content('b'),
    html.Br(),
    dbc.Row(
        [
            dbc.Button(
                "Reveal Correct Answer",
                id="reveal-correct-answer-button",
                color="primary",
            ),
            html.Br(),
            dbc.Collapse(
                id='correct-answer-collapse',
                children=[
                    html.P(id='correct-answer-text')
                ]
            )
        ]
    ),
    html.Br(),
    dbc.Row([
        dbc.Collapse(
            id='article-collapse',
            children=[
                html.H3(id='article-title'),
                html.P(id='article-text')
            ]
        ),
    ])
])

@app.callback(
    Output('correct-answer-collapse', 'is_open'),
    Output('correct-answer-text', 'children'),
    Output('reveal-correct-answer-button', 'n_clicks'),
    Input('reveal-correct-answer-button', 'n_clicks'),
    Input('article-dropdown', 'value'),
    Input('question-dropdown', 'value'),
)
def reveal_correct_answer(n_clicks, article_title, question_value):
    if not n_clicks or not article_title or not question_value:
        return False, "", 0

    article = [article for article in quality_data.unique_sets if article.title == article_title][0]
    questions = [q for q in article.questions]
    question_idx = [q.question for q in questions].index(question_value)
    correct_option_idx = questions[question_idx].gold_label - 1
    correct_option = questions[question_idx].options[correct_option_idx]
    return True, f"Correct Answer: {correct_option}", 0


@app.callback(
    Output('article-collapse', 'is_open'),
    Output('article-title', 'children'),
    Output('article-text', 'children'),
    Output('question-dropdown', 'options'),
    Output('question-dropdown', 'value'),
    Input('article-dropdown', 'value')
)
def display_article(article_title):
    if article_title is None:
        return False, "", "", [], ''

    article = [article for article in quality_data.unique_sets if article.title == article_title][0]
    if article is None:
        return False, "", "", []

    questions = [q for q in article.questions]
    question_options = [{'label': q.question, 'value': q.question} for q in questions]

    return True, article_title, article.article, question_options, ''

@app.callback(
    Output('option-a-text', 'children'),
    Output('option-b-text', 'children'),
    Input('article-dropdown', 'value'),
    Input('question-dropdown', 'value'),
)
def display_option_content(article_title, question_value):
    if not article_title or not question_value:
        return "", ""
    article = [article for article in quality_data.unique_sets if article.title == article_title][0]
    questions = [q for q in article.questions]
    question_idx = [q.question for q in questions].index(question_value)
    correct_option_idx = questions[question_idx].gold_label - 1
    distraction_option_idx = questions[question_idx].validation[0].untimed_best_distractor - 1
    sorted_option_indices = sorted([correct_option_idx, distraction_option_idx])
    return questions[question_idx].options[sorted_option_indices[0]], questions[question_idx].options[sorted_option_indices[1]]

def generate_argument_x(letter:str):
    @app.callback(
        Output(f'option-{letter}-unstructured-argument', 'value'),
        Input(f'generate-argument-{letter}-button', 'n_clicks'),
        State('article-dropdown', 'value'),
        State('question-dropdown', 'value'),
    )
    def _generate_argument_x(n_clicks, article_title, question_value):
        if n_clicks is None or article_title is None or question_value is None:
            return ""

        # Here you would call your Evaluator class to generate the argument
        # For demonstration, we'll just return a placeholder text
        article = [article for article in quality_data.unique_sets if article.title == article_title][0]
        questions = [q for q in article.questions]
        question_idx = [q.question for q in questions].index(question_value)
        correct_option_idx = questions[question_idx].gold_label - 1
        distraction_option_idx = questions[question_idx].validation[0].untimed_best_distractor - 1
        sorted_options = sorted([correct_option_idx, distraction_option_idx])
        if letter == 'a':
            option_idx = sorted_options[0]
        else:
            option_idx = sorted_options[1]
        evaluator = Evaluator(article, llm_config)
        generated_argument = evaluator.generate_argument(question_idx, option_idx)
        return generated_argument
    return _generate_argument_x

generate_argument_a = generate_argument_x('a')
generate_argument_b = generate_argument_x('b')

def evaluate_argument_x(letter:str):
    @app.callback(
        Output(f'evaluation-table-{letter}', 'children'),
        Input(f'review-evidence-{letter}-button', 'n_clicks'),
        State(f'option-{letter}-unstructured-argument', 'value'),
        State('article-dropdown', 'value'),
        State('question-dropdown', 'value'),
    )
    def _evaluate_argument_x(n_clicks, generated_argument, article_title, question_value):
        if n_clicks is None or not generated_argument:
            return html.P(
                "Type or generate an argument and click on the 'Generate Argument' button to review evidence.")
        # TODO: Handle when user clears article title or question value but there is still an argument
        article = [article for article in quality_data.unique_sets if article.title == article_title][0]
        questions = [q for q in article.questions]
        question_idx = [q.question for q in questions].index(question_value)

        evaluator = Evaluator(article, llm_config)
        structured_argument = evaluator.convert_text_to_structured_argument(generated_argument)
        evaluations = evaluator.evaluate_structured_argument(structured_argument)
        # Here you would call your Evaluator class to evaluate the argument
        # For demonstration, we'll just return a placeholder table
        table = dbc.Table.from_dataframe(
            pd.DataFrame({
                "Premises": [claim.text for claim in structured_argument.premises] + [
                    f"Conclusion: {structured_argument.conclusion.text}"],
                "Estimated Truth Value": [evaluation.truth_value for evaluation in evaluations],
                "Reason for Truth Value": [evaluation.reason_for_truth_value for evaluation in evaluations],
                "Supporting Sections": [
                    html.Ul([
                        html.Li(
                            html.A(
                                str(node_id),
                                href="#",
                                id={
                                    'type': f'option-{letter}-evidence-link',
                                    'index': node_id
                                }
                            )
                        )
                        for node_id in evaluation.source_node_ids
                    ])
                    for evaluation_idx, evaluation in enumerate(evaluations)
                ],
            }),
            striped=True,
            bordered=True,
            hover=True,
            size='sm'
        )

        return html.Div(
            table,
            style={
                'maxHeight': '400px',
                'overflowY': 'scroll',
                'border': '1px solid #ddd',
                'padding': '10px'
            }
        )

    return _evaluate_argument_x

evaluate_argument_a = evaluate_argument_x('a')
evaluate_argument_b = evaluate_argument_x('b')

def display_evidence_x(letter:str):
    @app.callback(
        Output(f'option-{letter}-evidence-review', 'value'),
        Input({'type': f'option-{letter}-evidence-link', 'index': ALL}, 'n_clicks'),
        State({'type': f'option-{letter}-evidence-link', 'index': ALL}, 'id'),
        State({'type': f'option-{letter}-evidence-link', 'index': ALL}, 'n_clicks_timestamp'),
        State('article-dropdown', 'value'),
    )
    def _display_evidence_x(n_clicks, ids, timestamps, article_title):
        if not any(n_clicks):
            return ""
        # Find the id of the most recently clicked link
        timestamps = [t if t is not None else -1 for t in timestamps]
        clicked_id = ids[timestamps.index(max(timestamps))]
        node_id = clicked_id['index']
        article = [article for article in quality_data.unique_sets if article.title == article_title][0]
        evaluator = Evaluator(article, llm_config)
        citation = evaluator.citation_engine.query("Hi!").source_nodes[node_id - 1].text

        return citation

    return _display_evidence_x

display_evidence_a = display_evidence_x('a')
display_evidence_b = display_evidence_x('b')
# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)