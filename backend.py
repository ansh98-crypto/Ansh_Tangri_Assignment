import os
from typing import TypedDict

from langgraph.graph import StateGraph, END

# LangChain OpenAI wrapper (recommended with LangGraph)
from langchain_openai import ChatOpenAI


class QAState(TypedDict):
    question: str
    answer: str


def _trim_to_words(text: str, max_words: int = 200) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]).rstrip() + "…"


def build_graph():
    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0.2,
        api_key=os.environ["OPENAI_API_KEY"],  # must exist
    )

    def answer_node(state: QAState) -> QAState:
        question = state["question"].strip()

        prompt = (
            "You are a helpful assistant. Answer the user's question clearly and directly.\n"
            "Hard constraint: keep the answer to 200 words or fewer.\n\n"
            f"User question: {question}"
        )

        resp = llm.invoke(prompt)
        text = getattr(resp, "content", str(resp))

        return {"question": question, "answer": _trim_to_words(text, 200)}

    g = StateGraph(QAState)
    g.add_node("answer", answer_node)
    g.set_entry_point("answer")
    g.add_edge("answer", END)

    return g.compile()


# Build once at import time (fast subsequent calls)
GRAPH = build_graph()


def run_langgraph(question: str) -> str:
    out = GRAPH.invoke({"question": question, "answer": ""})
    return out["answer"]
