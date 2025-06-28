from dash import Dash, html, dash_table, dcc, Input, Output, State
import webbrowser
import threading
import re
import time
import plotly.graph_objs as go

# List of stopwords to ignore common words
stopwords = set([
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves", "he", "him",
    "his", "himself", "she", "her", "hers", "herself", "it", "its",
    "itself", "they", "them", "their", "theirs", "themselves",
    "what", "which", "who", "whom", "this", "that", "these", "those",
    "am", "is", "are", "was", "were", "be", "been", "being", "have",
    "has", "had", "having", "do", "does", "did", "doing", "a", "an",
    "the", "and", "but", "if", "or", "because", "as", "until", "while",
    "of", "at", "by", "for", "with", "about", "against", "between",
    "into", "through", "during", "before", "after", "above", "below",
    "to", "from", "up", "down", "in", "out", "on", "off", "over", "under",
    "again", "further", "then", "once", "here", "there", "when", "where",
    "why", "how", "all", "any", "both", "each", "few", "more", "most",
    "other", "some", "such", "no", "nor", "not", "only", "own", "same",
    "so", "than", "too", "very", "s", "t", "can", "will", "just", "don",
    "should", "now"
])

def bold_common_words(input_text, ref_text):
    """
    Bolds the words in input_text that are also present in ref_text, excluding stopwords.

    Args:
        input_text: The text in which to bold common words.
        ref_text: The reference text to compare against.

    Returns:
        The input_text with common words bolded using Markdown syntax.
    """
    input_words = re.findall(r'\w+', input_text.lower())
    ref_words = re.findall(r'\w+', ref_text.lower())

    commons = set(input_words).intersection(ref_words).difference(stopwords)

    def replacer(match):
        word = match.group(0)
        if word.lower() in commons:
            return f"**{word}**"
        return word

    pattern = re.compile(r'\b\w+\b')
    return pattern.sub(replacer, input_text)

def interpolate_color(score):
    """
    Returns an RGBA color interpolated between light red and light green based on the given score.

    Args:
        score: A float between 0 and 1 used to interpolate the color.

    Returns:
        A string representing the interpolated RGBA color.
    """
    red = (255, 214, 214)   
    green = (182, 225, 160)  

    r = int(red[0] + (green[0] - red[0]) * score)
    g = int(red[1] + (green[1] - red[1]) * score)
    b = int(red[2] + (green[2] - red[2]) * score)

    return f'rgba({r},{g},{b},0.5)' 

def get_bleu_dashboard(all_results_grouped):
    """
    Get a Dash to visualize BLEU evaluation results.

    Args:
        all_results_grouped: A dictionary mapping run IDs to lists of BLEU evaluation results with metadata.

    The dashboard displays:
        - A table summarizing BLEU scores by test and average scores.
        - Detailed sentence-level BLEU scores with highlighted input and reference text.
        - Color-coded BLEU scores indicating performance.
    """
    app = Dash(__name__)

    title_to_runs = {}
    for run_id, results in all_results_grouped.items():
        for result in results:
            title = result.get('metadata', {}).get('title', 'Unknown')
            if title not in title_to_runs:
                title_to_runs[title] = []
            if run_id not in title_to_runs[title]:
                title_to_runs[title].append(run_id)

    titles = sorted(list(title_to_runs.keys()))

    title_options = [{'label': t, 'value': t} for t in titles]

    initial_title = titles[0] if titles else None
    initial_runs = title_to_runs.get(initial_title, []) if initial_title else []
    run_options = [{'label': f"{r}", 'value': r} for r in initial_runs]

    app.layout = html.Div([
        dcc.Location(id='url', refresh=True),
        html.H1("BLEU Evaluation Dashboard", style={'textAlign': 'center', 'marginBottom': '20px'}),

        dcc.Dropdown(
            id='title-dropdown',
            options=title_options,
            value=initial_title,
            clearable=False,
            style={'display': 'none'}
        ),
        dcc.Graph(
            id='bleu-bar-chart',
            style={'maxWidth': '900px', 'margin': 'auto', 'marginBottom': '40px'}
        ),
        dcc.Dropdown(
            id='run-dropdown',
            options=run_options,
            value=initial_runs[0] if initial_runs else None,
            clearable=False,
            style={'width': '400px', 'margin': 'auto', 'marginBottom': '20px'}
        ),

        dcc.Loading(
            id="loading-results",
            type="circle",
            fullscreen=False,
            children=html.Div(id='results-container', style={'maxWidth': '1300px', 'margin': 'auto', 'padding': '0 20px'})
        )
    ], style={'maxWidth': '1300px', 'margin': 'auto', 'padding': '40px 20px'})

    # Create a table with results for all tests
    @app.callback(
    Output('bleu-bar-chart', 'figure'),
    Input('title-dropdown', 'value')
    )
    def update_table(_): 
        all_run_ids = list(all_results_grouped.keys())

        fixed_titles = [
            "Spoofing", "Tampering", "Repudiation",
            "Information Disclosure", "Denial of Service", "Elevation of Privilege"
        ]

        data_matrix = []
        for run_id in all_run_ids:
            row = {"Test": run_id}
            total = 0
            count = 0
            
            best_reference_score = None

            for title in fixed_titles:
                score = None
                for res in all_results_grouped[run_id]:
                    if res.get('metadata', {}).get('title') == title:
                        score = res['results']['scores']['overall']
                        
                        sentence_scores = res['results']['scores'].get('sentence_level', [])
                        for s in sentence_scores:
                            sc = s.get("bleu_for_this_sentence", 0)
                            if best_reference_score is None or sc > best_reference_score:
                                best_reference_score = sc
                        break
                if score is not None:
                    row[title] = round(score, 4)
                    total += score
                    count += 1
                else:
                    row[title] = None

            row["Average"] = round(total / count, 4) if count else None
            row["Best Sentence-Lvl"] = round(best_reference_score, 4) if best_reference_score is not None else None
            data_matrix.append(row)

        header = ["Test"] + fixed_titles + ["Average", "Best Sentence-Lvl"]
        cells = [[row.get(col) for row in data_matrix] for col in header]

        table = go.Figure(data=[go.Table(
            header=dict(
                values=header,
                fill_color='#2c3e50',
                font=dict(color='white', size=14, family='Arial'),
                align='center'
            ),
            cells=dict(
                values=cells,
                fill_color='#f9f9f9',
                align='center',
                font=dict(color='#333', size=12, family='Arial'),
                height=30
            )
        )])

        table.update_layout(
            title='BLEU Scores by Test',
            margin=dict(l=20, r=20, t=60, b=20)
        )

        return table

    # Callback to update results tables for each test
    @app.callback(
        Output('results-container', 'children'),
        Input('run-dropdown', 'value')
    )
    def update_results(selected_run_id):
        time.sleep(0.5)
        if not selected_run_id:
            return "No data available."
        if selected_run_id not in all_results_grouped:
            return html.Div("Selected run not found.")
        results = all_results_grouped[selected_run_id]
        layout_children = []
        for result in results:
            title = result.get('metadata', {}).get('title', 'Unknown')
            scores = result['results']['scores']
            sentence_scores = scores['sentence_level']

            table_data = []
            for s in sentence_scores:
                input_highlighted = bold_common_words(s["input"], s["best_reference"])
                ref_highlighted = bold_common_words(s["best_reference"], s["input"])
                table_data.append({
                    "Input": input_highlighted,
                    "Best Reference": ref_highlighted,
                    "BLEU Score": round(s["bleu_for_this_sentence"], 4)
                })

            style_data_conditional = []
            for i, row in enumerate(table_data):
                score = row["BLEU Score"]
                bgcolor = interpolate_color(score)
                text_color = '#2c3e50' if score < 0.5 else '#1a3d12'
                style_data_conditional.append({
                    'if': {'row_index': i},
                    'backgroundColor': bgcolor,
                    'color': text_color,
                    'fontWeight': '400'
                })

            layout_children += [
                html.H2(f"{title} - Overall BLEU Score: {round(scores['overall'], 4)}",
                        style={'marginTop': '40px', 'marginBottom': '20px'}),
                html.H3("Sentence-Level BLEU Scores", style={'marginBottom': '15px'}),
                dash_table.DataTable(
                    columns=[
                        {"name": "Input", "id": "Input", 'presentation': 'markdown'},
                        {"name": "Best Reference", "id": "Best Reference", 'presentation': 'markdown'},
                        {"name": "BLEU Score", "id": "BLEU Score"}
                    ],
                    data=table_data,
                    page_action='none',
                    style_table={
                        'width': '100%',
                        'overflowX': 'visible',
                        'overflowY': 'visible',
                        'minWidth': '100%'
                    },
                    style_cell={
                        'whiteSpace': 'normal',
                        'height': 'auto',
                        'textAlign': 'justify',
                        'padding': '10px',
                        'fontFamily': 'Segoe UI, Tahoma, Geneva, Verdana, sans-serif',
                        'fontSize': '15px',
                        'lineHeight': '1.4'
                    },
                    style_cell_conditional=[
                        {'if': {'column_id': 'BLEU Score'}, 'width': '12%', 'textAlign': 'center', 'fontWeight': '700'},
                        {'if': {'column_id': 'Input'}, 'width': '44%'},
                        {'if': {'column_id': 'Best Reference'}, 'width': '44%'},
                    ],
                    style_header={
                        'backgroundColor': '#f7f9fc',
                        'fontWeight': '700',
                        'borderBottom': '2px solid #ccc',
                        'fontSize': '16px',
                    },
                    style_data_conditional=style_data_conditional,
                    style_as_list_view=True,
                    cell_selectable=False
                ),
                html.Hr(style={'marginTop': '40px', 'marginBottom': '40px'})
            ]
        return layout_children
    
    return app

# Dasboard creation
def launch_bleu_dashboard(all_results_grouped):
    app = get_bleu_dashboard(all_results_grouped)

    threading.Timer(1.0, lambda: webbrowser.open("http://127.0.0.1:8050")).start()
    app.run(debug=False)