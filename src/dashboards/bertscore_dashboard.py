from dash import Dash, html, dash_table, dcc, Input, Output, State
import webbrowser
import threading
import re
import time
import plotly.graph_objs as go

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

# Dasboard creation
def launch_bertscore_dashboard(all_results_grouped):
    """
    Launches a Dash web application to visualize Bert Score evaluation results.

    Args:
        all_results_grouped: A dictionary mapping run IDs to lists of Bert Score evaluation results with metadata.

    The dashboard displays:
        - A table summarizing Bert Score scores by test and average scores.
        - Detailed sentence-level Bert Score scores
        - Color-coded Bert Score scores indicating performance.
    """
    app = Dash(__name__)

    run_ids = list(all_results_grouped.keys())

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
        html.H1("BERTSCORE Evaluation Dashboard", style={'textAlign': 'center', 'marginBottom': '20px'}),

        dcc.Dropdown(
            id='title-dropdown',
            options=title_options,
            value=initial_title,
            clearable=False,
            style={'display': 'none'}
        ),
        dcc.Graph(
            id='bertscore-bar-chart',
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
    Output('bertscore-bar-chart', 'figure'),
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
                        score = res['results']['scores']['overall']['f1']
                        
                        sentence_scores = res['results']['scores'].get('sentence_level', [])
                        for s in sentence_scores:
                            sc = s.get("f1", 0)
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
            title='BERTSCORE F1 Scores by Test',
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

        first_result = results[0]
        model = first_result.get('results', {}).get('model', 'Unknown')

        layout_children.append(
            html.H4(f"Model used for the selected test: {model}",
                    style={'textAlign': 'center', 'marginBottom': '20px'})
        )
        for result in results:
            title = result.get('metadata', {}).get('title', 'Unknown')
            model = result.get('results', {}).get('model', 'Unknown')
            scores = result['results']['scores']
            sentence_scores = scores['sentence_level']

            table_data = []
            for s in sentence_scores:
                table_data.append({
                    "Input": s["input"],
                    "Best Reference": s["best_reference"],
                    "BERTSCORE F1": round(s["f1"], 4)
                })

            style_data_conditional = []
            for i, row in enumerate(table_data):
                score = row["BERTSCORE F1"]
                bgcolor = interpolate_color(score)
                text_color = '#2c3e50' if score < 0.5 else '#1a3d12'
                style_data_conditional.append({
                    'if': {'row_index': i},
                    'backgroundColor': bgcolor,
                    'color': text_color,
                    'fontWeight': '400'
                })
            layout_children += [
                html.H2(f"{title} - Overall BERTSCORE F1 Score: {round(scores['overall']['f1'], 4)}",
                        style={'marginTop': '40px', 'marginBottom': '20px'}),
                html.H3("Sentence-Level BERTSCORE F1 Scores", style={'marginBottom': '15px'}),
                dash_table.DataTable(
                    columns=[
                        {"name": "Input", "id": "Input", 'presentation': 'markdown'},
                        {"name": "Best Reference", "id": "Best Reference", 'presentation': 'markdown'},
                        {"name": "BERTSCORE F1", "id": "BERTSCORE F1"}
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
                        {'if': {'column_id': 'BERTSCORE F1'}, 'width': '12%', 'textAlign': 'center', 'fontWeight': '700'},
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

    threading.Timer(1.0, lambda: webbrowser.open("http://127.0.0.1:8050")).start()
    app.run(debug=False)