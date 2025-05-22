from dash import Dash, html, dash_table, dcc, Input, Output, State
import webbrowser
import threading
import time
import plotly.graph_objs as go

# Dasboard creation
def launch_dimension_dashboard(all_results_grouped):
    """
    Launches a Dash web application to visualize LLM-as-a-Judge evaluation results.

    Args:
        all_results_grouped: A dictionary mapping run IDs to lists of LLM-as-a-Judge evaluation results with metadata.

    The dashboard displays:
        - A table summarizing LLM-as-a-Judge average scores and final score by test.
        - Spider diagram by test.
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
        html.H1("LLM-as-a-Judge Evaluation Dashboard", style={'textAlign': 'center', 'marginBottom': '20px'}),

        dcc.Dropdown(
            id='title-dropdown',
            options=title_options,
            value=initial_title,
            clearable=False,
            style={'display': 'none'}
        ),
        dcc.Graph(
            id='dimension-bar-chart',
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
        ),
        dcc.Graph(
            id='dimension-radar-chart',
            style={'maxWidth': '600px', 'margin': 'auto', 'marginTop': '40px'}
        )
    ], style={'maxWidth': '1300px', 'margin': 'auto', 'padding': '40px 20px'})

    # Create a table with results for all tests
    @app.callback(
        Output('dimension-bar-chart', 'figure'),
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

            total1 = total2 = total3 = total4 = totalf = 0
            count1 = count2 = count3 = count4 = countf = 0

            for title in fixed_titles:
                score1 = score2 = score3 = score4 = scoref = None

                for res in all_results_grouped[run_id]:
                    if res.get('metadata', {}).get('title') == title:
                        score1 = res['results']['dimensions']['consistency']['average']
                        score2 = res['results']['dimensions']['coverage']['average']
                        score3 = res['results']['dimensions']['plausibility']['average']
                        score4 = res['results']['dimensions']['relevance']['average']
                        scoref = res['results']['dimensions']['score']

                        break

                if score1 is not None:
                    total1 += score1
                    count1 += 1
                if score2 is not None:
                    total2 += score2
                    count2 += 1
                if score3 is not None:
                    total3 += score3
                    count3 += 1
                if score4 is not None:
                    total4 += score4
                    count4 += 1
                if scoref is not None:
                    totalf += scoref
                    countf += 1

            row["Average Consistency"] = round(total1 / count1, 4) if count1 else None
            row["Average Coverage"] = round(total2 / count2, 4) if count2 else None
            row["Average Plausibility"] = round(total3 / count3, 4) if count3 else None
            row["Average Relevance"] = round(total4 / count4, 4) if count4 else None
            row["Average Score"] = round(totalf / countf, 4) if countf else None

            data_matrix.append(row)

        header = [
            "Test", 
            "Average Consistency", "Average Coverage", "Average Plausibility", "Average Relevance", "Average Score" 
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
            title='Dimensions Average Scores by Test',
            margin=dict(l=20, r=20, t=60, b=20)
        )

        return table
    
    # Callback to show llm informations and spider for each test
    @app.callback(
        Output('results-container', 'children'),
        Output('dimension-radar-chart', 'figure'),
        Input('run-dropdown', 'value')
    )
    def update_results(selected_run_id):
        time.sleep(0.5)

        if not selected_run_id:
            return html.Div("No data available."), go.Figure()

        if selected_run_id not in all_results_grouped:
            return html.Div("Selected run not found."), go.Figure()

        results = all_results_grouped[selected_run_id]
        layout_children = []

        first_result = results[0]
        llm = first_result.get('results', {}).get('llm', 'Unknown')
        num_completions = first_result.get('results', {}).get('num_completions', 'Unknown')

        layout_children.extend([
            html.H4(f"LLM acted as a Judge for the selected test: {llm}",
                    style={'textAlign': 'center', 'marginBottom': '20px'}),
            html.H4(f"Number of times the LLM evaluated each dimension within each STRIDE category: {num_completions}",
                    style={'textAlign': 'center', 'marginBottom': '20px'})
        ])

        categories = ['consistency', 'coverage', 'plausibility', 'relevance']
        scores = []

        for dim in categories:
            total = 0
            count = 0
            for r in results:
                try:
                    value = r['results']['dimensions'][dim]['average']
                    total += value
                    count += 1
                except KeyError:
                    pass
            avg = round(total / count, 4) if count else 0
            scores.append(avg)

        scores.append(scores[0])
        categories.append(categories[0])

        radar_fig = go.Figure()

        radar_fig.add_trace(go.Scatterpolar(
            r=scores,
            theta=categories,
            fill='toself',
            name='Average Scores'
        ))

        radar_fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[1, 5]
                )),
            showlegend=False,
        )

        return layout_children, radar_fig

    threading.Timer(1.0, lambda: webbrowser.open("http://127.0.0.1:8050")).start()
    app.run(debug=False)