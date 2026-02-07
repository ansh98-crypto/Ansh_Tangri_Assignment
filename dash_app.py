import os
from dash import Dash, html, dcc, Input, Output, State

from backend import run_langgraph

app = Dash(__name__)
server = app.server  # <-- for gunicorn on Heroku: dash_app:server

app.layout = html.Div(
    style={"maxWidth": "900px", "margin": "40px auto", "fontFamily": "Arial"},
    children=[
        html.H2("LangGraph Q&A (≤200 words)"),
        dcc.Textarea(
            id="question",
            placeholder="Type your question here…",
            style={"width": "100%", "height": 120},
        ),
        html.Button("Ask", id="ask", n_clicks=0, style={"marginTop": 12}),
        html.Hr(),
        html.Div("Answer:", style={"fontWeight": "bold", "marginBottom": 8}),
        html.Pre(
            id="answer",
            style={
                "whiteSpace": "pre-wrap",
                "background": "#f6f6f6",
                "padding": "12px",
                "borderRadius": "8px",
            },
        ),
        html.Div(id="status", style={"marginTop": 8, "fontSize": "12px", "color": "#555"}),
    ],
)

@app.callback(
    Output("answer", "children"),
    Output("status", "children"),
    Input("ask", "n_clicks"),
    State("question", "value"),
    prevent_initial_call=True,
)
def ask(n_clicks, question):
    if not question or not question.strip():
        return "Please type a question.", ""

    try:
        ans = run_langgraph(question)
        word_count = len(ans.split())
        return ans, f"Returned {word_count} words (max 200)."
    except Exception as e:
        return "Error while generating answer. Check logs.", f"Error: {e}"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8050"))
    app.run(host="0.0.0.0", port=port)
