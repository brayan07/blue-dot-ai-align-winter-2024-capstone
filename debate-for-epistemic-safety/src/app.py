import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import json
import os
from debate_for_epistemic_safety.pipelines.master.nodes import QualityData

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Load your articles and questions data
with open('data/01_raw/quality_filtered_train.json') as f:
    data = json.load(f)
    quality_data = QualityData(**data)

# Extract article titles for the dropdown
article_titles = [article.title for article in quality_data.articles]

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
    dbc.Row([
        dbc.Collapse(
            id='article-collapse',
            children=[
                html.H3(id='article-title'),
                html.P(id='article-text')
            ]
        ),
        dbc.Collapse(
            id='question-section',
            children=[
                dcc.Textarea(
                    id='argument-input',
                    placeholder="Enter your argument here...",
                    style={'width': '100%', 'height': 200}
                ),
                dbc.Button("Generate Argument", id='generate-argument-button', color='primary'),
                html.Div(id='generated-argument'),
                html.Div(id='evaluation-table')
            ]
        )
    ])
])


@app.callback(
    Output('article-collapse', 'is_open'),
    Output('article-title', 'children'),
    Output('article-text', 'children'),
    Output('question-dropdown', 'options'),
    Input('article-dropdown', 'value')
)
def display_article(article_title):
    if article_title is None:
        return False, "", "", []

    article = [article for article in quality_data.articles if article.title == article_title ][0]
    if article is None:
        return False, "", "", []

    questions = [q for q in article.questions]
    question_options = [{'label': q.question, 'value': q.question} for q in questions]

    return True, article_title, article.article, question_options



@app.callback(
    Output('generated-argument', 'children'),
    Input('generate-argument-button', 'n_clicks'),
    State('article-dropdown', 'value'),
    State('question-dropdown', 'value'),
    State('argument-input', 'value')
)
def generate_argument(n_clicks, article_title, question, argument_text):
    if n_clicks is None or article_title is None or question is None:
        return ""

    # Here you would call your Evaluator class to generate the argument
    # For demonstration, we'll just return a placeholder text
    generated_argument = "Generated argument based on the selected question and article."

    return generated_argument


@app.callback(
    Output('evaluation-table', 'children'),
    Input('generated-argument', 'children')
)
def evaluate_argument(generated_argument):
    if not generated_argument:
        return ""

    # Here you would call your Evaluator class to evaluate the argument
    # For demonstration, we'll just return a placeholder table
    table = dbc.Table.from_dataframe(
        pd.DataFrame({
            "Premise": ["Premise 1", "Premise 2"],
            "Truth Value": ["LIKELY_TRUE", "UNCERTAIN"],
            "Supporting Sections": ["Section 1", "Section 2"]
        }),
        striped=True,
        bordered=True,
        hover=True
    )

    return table
# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)