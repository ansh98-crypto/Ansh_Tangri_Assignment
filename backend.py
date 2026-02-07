import os
from typing import TypedDict
from langgraph.graph import StateGraph, END
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


class QAState(TypedDict):
    question: str
    answer: str


def trim_words(text: str, limit=200):
    words = text.split()
    return " ".join(words[:limit])


def answer_node(state: QAState) -> QAState:
    question = state["question"]

    prompt = f"""
Answer the following question clearly in UNDER 200 words.

Question:
{question}
"""

    resp = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
    )

    text = resp.choices[0].message.content

    return {
        "question": question,
        "answer": trim_words(text, 200),
    }


graph = StateGraph(QAState)
graph.add_node("answer", answer_node)
graph.set_entry_point("answer")
graph.add_edge("answer", END)

GRAPH = graph.compile()


def run_langgraph(question: str) -> str:
    out = GRAPH.invoke({"question": question, "answer": ""})
    return out["answer"]
