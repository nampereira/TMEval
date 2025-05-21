from dash import Dash, html, dash_table, dcc, Input, Output, State
import webbrowser
import threading
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
def launch_rouge_dashboard(all_results_grouped):
    """
    Launches a Dash web application to visualize ROUGE evaluation results.

    Args:
        all_results_grouped: A dictionary mapping run IDs to lists of ROUGE evaluation results with metadata.

    The dashboard displays:
        - A table summarizing ROUGE scores by test.
        - Detailed sentence-level ROUGE scores.
        - Color-coded ROUGE scores indicating performance.
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
    run_options = [{'label': f"Test {r[:8]}...", 'value': r} for r in initial_runs]

    app.layout = html.Div([
        dcc.Location(id='url', refresh=True),
        html.H1("ROUGE Evaluation Dashboard", style={'textAlign': 'center', 'marginBottom': '20px'}),

        dcc.Dropdown(
            id='title-dropdown',
            options=title_options,
            value=initial_title,
            clearable=False,
            style={'display': 'none'}
        ),
        dcc.Graph(
            id='rouge-bar-chart',
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
        Output('rouge-bar-chart', 'figure'),
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
            row = {"Test": run_id[:8] + "..."}

            total1 = total2 = totall = 0
            count1 = count2 = countl = 0

            best_rouge1 = 0
            best_rouge2 = 0
            best_rougeL = 0

            for title in fixed_titles:
                score1 = score2 = scorel = None

                for res in all_results_grouped[run_id]:
                    if res.get('metadata', {}).get('title') == title:
                        score1 = res['results']['scores']['overall']['rouge1']['fmeasure']
                        score2 = res['results']['scores']['overall']['rouge2']['fmeasure']
                        scorel = res['results']['scores']['overall']['rougeL']['fmeasure']

                        sentence_scores = res['results']['scores'].get('sentence_level', [])
                        for s in sentence_scores:
                            r1 = s.get("rouge_for_this_sentence", {}).get("rouge1", {}).get("fmeasure", 0)
                            r2 = s.get("rouge_for_this_sentence", {}).get("rouge2", {}).get("fmeasure", 0)
                            rl = s.get("rouge_for_this_sentence", {}).get("rougeL", {}).get("fmeasure", 0)
                            best_rouge1 = max(best_rouge1, r1)
                            best_rouge2 = max(best_rouge2, r2)
                            best_rougeL = max(best_rougeL, rl)

                        break

                if score1 is not None:
                    total1 += score1
                    count1 += 1
                if score2 is not None:
                    total2 += score2
                    count2 += 1
                if scorel is not None:
                    totall += scorel
                    countl += 1

            row["Average Rouge1"] = round(total1 / count1, 4) if count1 else None
            row["Average Rouge2"] = round(total2 / count2, 4) if count2 else None
            row["Average RougeL"] = round(totall / countl, 4) if countl else None

            row["Best Sentence-Lvl ROUGE-1"] = round(best_rouge1, 4) if best_rouge1 > 0 else None
            row["Best Sentence-Lvl ROUGE-2"] = round(best_rouge2, 4) if best_rouge2 > 0 else None
            row["Best Sentence-Lvl ROUGE-L"] = round(best_rougeL, 4) if best_rougeL > 0 else None

            data_matrix.append(row)

        header = [
            "Test", 
            "Average Rouge1", "Average Rouge2", "Average RougeL",
            "Best Sentence-Lvl ROUGE-1", "Best Sentence-Lvl ROUGE-2", "Best Sentence-Lvl ROUGE-L"
        ]

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
            title='ROUGE F1 Scores by Test',
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
            sentence_scores = scores.get('sentence_level', [])

            table_data = []
            for s in sentence_scores:

                input = s["input"]
                rouge1_f1 = s.get("rouge_for_this_sentence", {}).get("rouge1", {}).get("fmeasure", None)
                rouge2_f1 = s.get("rouge_for_this_sentence", {}).get("rouge2", {}).get("fmeasure", None)
                rougeL_f1 = s.get("rouge_for_this_sentence", {}).get("rougeL", {}).get("fmeasure", None)

                table_data.append({
                    "Input": input,
                    "ROUGE-1 F1": round(rouge1_f1, 4) if rouge1_f1 is not None else None,
                    "ROUGE-2 F1": round(rouge2_f1, 4) if rouge2_f1 is not None else None,
                    "ROUGE-L F1": round(rougeL_f1, 4) if rougeL_f1 is not None else None
                })

            style_data_conditional = []
            for i, row in enumerate(table_data):
                for col in ["ROUGE-1 F1", "ROUGE-2 F1", "ROUGE-L F1"]:
                    score = row[col]
                    bgcolor = interpolate_color(score)
                    text_color = '#2c3e50' if score < 0.5 else '#1a3d12'
                    style_data_conditional.append({
                        'if': {'row_index': i, 'column_id': col},
                        'backgroundColor': bgcolor,
                        'color': text_color,
                        'fontWeight': '400'
                    })


            layout_children += [
                html.H2(f"{title}",
                        style={'marginTop': '40px', 'marginBottom': '20px'}),
                html.H3("Sentence-Level ROUGE Scores for Best References", style={'marginBottom': '15px'}),
                dash_table.DataTable(
                    columns=[
                        {"name": "Input", "id": "Input", 'presentation': 'markdown'},
                        {"name": "ROUGE-1 F1", "id": "ROUGE-1 F1"},
                        {"name": "ROUGE-2 F1", "id": "ROUGE-2 F1"},
                        {"name": "ROUGE-L F1", "id": "ROUGE-L F1"}
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
                        {'if': {'column_id': 'ROUGE-1 F1'}, 'width': '18%', 'textAlign': 'center', 'fontWeight': '700'},
                        {'if': {'column_id': 'ROUGE-2 F1'}, 'width': '18%', 'textAlign': 'center', 'fontWeight': '700'},
                        {'if': {'column_id': 'ROUGE-L F1'}, 'width': '18%', 'textAlign': 'center', 'fontWeight': '700'},
                        {'if': {'column_id': 'Input'}, 'width': '46%'},
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