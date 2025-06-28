from dash import Dash, dcc, html, Output, Input
import webbrowser
import threading
import time

from src.dashboards.bleu_dashboard import get_bleu_dashboard
from src.dashboards.rouge_dashboard import get_rouge_dashboard

def launch_multi_dashboard(all_results):
    dashboard_mapping = {
        'bleu': get_bleu_dashboard,
        'rouge': get_rouge_dashboard,
        # adicione outros se quiser
    }

    # # Extrai todos os resultados individuais (listas dentro do dict)
    # all_individual_results = []
    # for results_list in all_results.values():
    #     all_individual_results.extend(results_list)

    # Obtém os dashboards disponíveis filtrando pelo evaluator
    # Ajuste a chave usada para identificar o evaluator; no seu caso parece ser:
    # r['results']['evaluators_used'] contém a lista de evaluators usados, e
    # cada evaluator pode ser 'ROUGEEvaluator', 'BLEUEvaluator', etc.
    # Vamos mapear para chaves minúsculas que usamos no dashboard_mapping

    available_dashboards = {}
    for key, func in dashboard_mapping.items():
        # Verifica se algum resultado tem esse evaluator (ignorando case)
        found = False
        for r in all_results:
            evaluators_used = r.get('results', {}).get('evaluators_used', [])
            # Normalize os nomes para lower e sem "Evaluator" para comparar
            normalized = [e.lower().replace('evaluator', '') for e in evaluators_used]
            if key in normalized:
                found = True
                break
        if found:
            available_dashboards[key] = func

    if not available_dashboards:
        print("Nenhum dashboard disponível para os evaluators encontrados.")
        return

    app = Dash(__name__)

    app.layout = html.Div([
        html.H1("Multi-Evaluator Dashboard"),
        dcc.Tabs(
            id="tabs",
            value=list(available_dashboards.keys())[0],  # primeira tab
            children=[
                dcc.Tab(label=key.upper(), value=key)
                for key in available_dashboards
            ]
        ),
        html.Div(id='tab-content')
    ])

    @app.callback(
        Output('tab-content', 'children'),
        Input('tabs', 'value')
    )
    def render_tab(tab_name):
        if tab_name not in available_dashboards:
            return html.Div("Dashboard não disponível.")

        filtered_results = []
        for r in all_results:
            # Acessa o dicionário com os resultados por evaluator
            results_per_evaluator = r.get('results', {}).get('results', {}).get(tab_name)
            # Verifica se o evaluator/tab existe nos resultados
            if tab_name in results_per_evaluator:
                filtered_results.append(r)

        # Passa só os resultados que têm o evaluator/tab correspondente
        print(filtered_results)
        return available_dashboards[tab_name](filtered_results)

    # Abre o navegador automaticamente
    def open_browser():
        time.sleep(1)
        webbrowser.open_new("http://127.0.0.1:8050/")

    threading.Thread(target=open_browser).start()
    app.run(debug=False)
