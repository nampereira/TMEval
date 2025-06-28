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
    run_options = [{'label': f"{r}", 'value': r} for r in initial_runs]

    app.layout = html.Div([
        dcc.Location(id='url', refresh=True),
        html.H1("LLM-as-a-Judge Evaluation Dashboard", style={'textAlign': 'center', 'marginBottom': '20px'}),

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
    ], style={'maxWidth': '1300px', 'margin': 'auto', 'padding': '40px 20px'})
    
    # Callback to show llm informations and spider for each category for each test
    @app.callback(
        Output('results-container', 'children'),
        Input('run-dropdown', 'value')
    )
    def update_results(selected_run_id):
        time.sleep(0.5)

        if not selected_run_id or selected_run_id not in all_results_grouped:
            return html.Div("No data available.")

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

        common_dims = ['consistency', 'plausibility']

        specific_dims_by_category = {
            "Denial of Service": [
                'attack-types-coverage',
                'dos-protection-gaps-coverage',
                'resource-coverage',
            ],
            "Elevation of Privilege": [
                'control-gaps-coverage',
                'exploit-methods-coverage',
                'vulnerability-point-coverage',
            ],
            "Information Disclosure": [
                'attack-methods-coverage',
                'data-coverage',
                'id-protection-gaps-coverage',
            ],
            "Repudiation": [
                'action-coverage',
                'rep-attack-vectors-coverage',
                'logging-gaps-coverage',
            ],
            "Spoofing": [
                'spo-attack-vectors-coverage',
                'authentication-gaps-coverage',
                'entity-coverage',
            ],
            "Tampering": [
                'asset-coverage',
                'integrity-gaps-coverage',
                'tampering-methods-coverage',
            ],
        }

        titles = ["Spoofing", "Tampering", "Repudiation", "Information Disclosure", "Denial of Service", "Elevation of Privilege"]

        for title in titles:
            dims_to_show = common_dims + specific_dims_by_category.get(title, [])

            scores = []
            for dim in dims_to_show:
                total = 0
                count = 0
                for r in results:
                    if r.get('metadata', {}).get('title') == title:
                        try:
                            value = r['results']['dimensions'][dim]['average']
                            total += value
                            count += 1
                        except KeyError:
                            pass
                avg = round(total / count, 4) if count else 0
                scores.append(avg)

            scores.append(scores[0])
            theta = dims_to_show + [dims_to_show[0]]

            radar_fig = go.Figure()
            radar_fig.add_trace(go.Scatterpolar(
                r=scores,
                theta=theta,
                fill='toself',
                name=title
            ))

            radar_fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[1, 5])),
                showlegend=False,
                title=title
            )

            layout_children.append(dcc.Graph(figure=radar_fig, style={'maxWidth': '600px', 'margin': '40px auto'}))

            final_score = None
            for r in results:
                if r.get('metadata', {}).get('title') == title:
                    final_score = r['results']['dimensions'].get('score')
                    break

            if final_score is not None:
                layout_children.append(html.H5(
                    f"Final Score: {round(final_score, 4)}",
                    style={'textAlign': 'center', 'marginBottom': '40px', 'color': '#2c3e50'}
                ))

        return layout_children

    threading.Timer(1.0, lambda: webbrowser.open("http://127.0.0.1:8050")).start()
    app.run(debug=False)