import os
import streamlit as st
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# Streamlit page config + neon theme (UI layer only — nothing
# below this changes the classifier / RAG / graph logic)
# ============================================================

st.set_page_config(
    page_title="CampusOS",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'JetBrains Mono', monospace;
    }

    .stApp {
        background: radial-gradient(circle at 15% 10%, #1a0b3d 0%, #0a0118 45%, #05010f 100%);
        color: #e8e6f5;
    }

    section[data-testid="stSidebar"] {
        background: #0c0621;
        border-right: 1px solid rgba(157, 78, 221, 0.4);
    }

    .neon-title {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        font-size: 2.6rem;
        color: #ffffff;
        text-shadow: 0 0 6px #00f5ff, 0 0 18px #00f5ff, 0 0 42px rgba(0, 245, 255, 0.4);
        margin-bottom: 0.1rem;
        letter-spacing: 0.5px;
    }

    .neon-subtitle {
        color: #b9a9e0;
        font-size: 0.95rem;
        margin-top: 0;
        margin-bottom: 1.4rem;
    }

    .sidebar-heading {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        font-size: 1.05rem;
        color: #ff2ec4;
        text-shadow: 0 0 10px rgba(255, 46, 196, 0.5);
        margin-bottom: 0.6rem;
    }

    .programme-card {
        border: 1px solid rgba(0, 245, 255, 0.35);
        border-radius: 12px;
        padding: 0.8rem 1rem;
        margin-bottom: 1.2rem;
        background: rgba(0, 245, 255, 0.04);
        font-size: 0.85rem;
        color: #cfe9ff;
    }

    [data-testid="stChatMessage"] {
        border-radius: 14px;
        padding: 0.35rem 0.9rem;
        margin-bottom: 0.6rem;
        background: rgba(255, 255, 255, 0.02);
    }

    [data-testid="stChatMessage"]:nth-of-type(odd) {
        border-left: 3px solid #00f5ff;
        box-shadow: -4px 0 16px -6px rgba(0, 245, 255, 0.55);
    }

    [data-testid="stChatMessage"]:nth-of-type(even) {
        border-left: 3px solid #ff2ec4;
        box-shadow: -4px 0 16px -6px rgba(255, 46, 196, 0.55);
    }

    [data-testid="stChatInput"] textarea {
        background: #100a24 !important;
        color: #f1eefc !important;
        border: 1px solid rgba(0, 245, 255, 0.5) !important;
        border-radius: 10px !important;
    }

    [data-testid="stChatInput"]:focus-within {
        box-shadow: 0 0 16px rgba(0, 245, 255, 0.45);
    }

    .stButton > button {
        background: transparent;
        border: 1px solid #9d4edd;
        color: #d9c8ff;
        border-radius: 8px;
        transition: 0.2s ease-in-out;
    }

    .stButton > button:hover {
        box-shadow: 0 0 14px rgba(157, 78, 221, 0.7);
        border-color: #ff2ec4;
        color: #ffffff;
    }

    div[role="radiogroup"] label {
        color: #e8e6f5;
    }

    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-thumb { background: #9d4edd; border-radius: 4px; }
    ::-webkit-scrollbar-track { background: #0a0118; }
    </style>
    """,
    unsafe_allow_html=True,
)

#Step 1 - Building the RAG retrievers

def build_retriver(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    document = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=800,
                                              chunk_overlap=100)

    chunks = splitter.split_documents(document)

    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore.as_retriever(search_kwargs={"k": 4})

#step2 - State

class State(TypedDict):
    programme : str
    messages : Annotated[list,add_messages]
    query_type : str
    retrieved_context : str

#Step 3 - Nodes generation

def classifier_node(state : State) -> dict:
    """Look at the latest user message and decide which path to take."""

    last_message = state['messages'][-1].content

    prompt = (
        "Classify the following student query into exactly one category: "
        "'academic', 'fee', or 'general'.\n\n"
        "Use 'academic' for questions about attendance, exams, grading, credits, "
        "promotion, course structure, summer training, or degree requirements.\n"
        "Use 'fee' for questions about tuition, payment, refund, late charges, "
        "scholarships, or any money-related topic.\n"
        "Use 'general' for greetings, casual talk, or anything not related to "
        "the college rules or fee.\n\n"
        f"Query: {last_message}\n\n"
        "Return only one word: academic, fee, or general."
    )

    response = llm.invoke(prompt)
    category = response.content.strip().lower()

    if "academic" in category:
        category = "academic"
    elif "fee" in category:
        category = "fee"
    else:
        category = "general"

    return {"query_type" : category}

def academic_rag_node(state: State) -> dict:
    """Retrieves relevant chunks from the academics handbook."""
    query = state["messages"][-1].content
    docs = acedemic_retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    return {"retrieved_context": context}

def fee_rag_node(state: State) -> dict:
    """Retrieves relevant chunks from the fee structure PDF."""
    query = state["messages"][-1].content
    docs = fee_retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    return {"retrieved_context": context}


def general_node(state: State) -> dict:
    """Answers directly using the LLM's own knowledge, no retrieval needed."""
    return {"retrieved_context": "NO_RETRIEVAL_NEEDED"}


def response_node(state: State) -> dict:
    """Generates the final answer, personalized using the student's programme."""
    query = state["messages"][-1].content
    programme = state.get("programme", "Unknown")
    context = state["retrieved_context"]

    if context == "NO_RETRIEVAL_NEEDED":
        prompt = (
            f"You are a friendly college assistant talking to a {programme} student. "
            f"Answer this question using your own general knowledge:\n\n{query}"
        )
    else:
        prompt = (
            f"You are a college assistant helping a {programme} student. "
            f"Use the following context from the official college documents to answer "
            f"the question accurately. If the context mentions specific figures for "
            f"different programmes, highlight the one relevant to {programme} if possible.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Give a clear, friendly, and precise answer."
        )

    response = llm.invoke(prompt)
    return {"messages": [("ai", response.content.strip())]}


#step 4 - router function

def route_query(state:State):
    if state['query_type'] == 'academic':
        return "academic_rag"
    elif state['query_type'] == "fee":
        return "fee_rag"
    else:
        return "general"


# ============================================================
# One-time backend setup, cached so Streamlit doesn't rebuild
# the embeddings model / FAISS indexes / graph on every rerun.
# The graph structure and nodes above are exactly as written —
# this just controls *when* they get built.
# ============================================================

@st.cache_resource(show_spinner="Loading knowledge base and models (first run only)...")
def initialize_backend():
    global embeddings, acedemic_retriever, fee_retriever, llm

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    acedemic_retriever = build_retriver("academics_handbook.pdf")
    fee_retriever = build_retriver("fee_structure.pdf")

    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.4)

    #step 5 - Building the graph

    graph = StateGraph(State)

    graph.add_node("classifier",classifier_node)
    graph.add_node("academic_rag",academic_rag_node)
    graph.add_node("fee_rag",fee_rag_node)
    graph.add_node("general",general_node)
    graph.add_node("response",response_node)

    #edges

    graph.add_edge(START,"classifier")

    graph.add_conditional_edges(
        "classifier",route_query
    )

    graph.add_edge("academic_rag","response")
    graph.add_edge("fee_rag","response")
    graph.add_edge("general","response")

    graph.add_edge("response",END)

    return graph.compile()


app = initialize_backend()

#step 6 - Streamlit UI (replaces the original console input()/print() loop)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of (role, text) tuples

if "programme" not in st.session_state:
    st.session_state.programme = "BCA"

with st.sidebar:
    st.markdown("<div class='sidebar-heading'>Your profile</div>", unsafe_allow_html=True)

    st.session_state.programme = st.radio(
        "Which programme are you in?",
        ["BCA", "BBA", "B.Com (H)"],
        index=["BCA", "BBA", "B.Com (H)"].index(st.session_state.programme),
    )

    st.markdown(
        f"<div class='programme-card'>Answers will be personalized for a "
        f"<strong>{st.session_state.programme}</strong> student.</div>",
        unsafe_allow_html=True,
    )

    if st.button("Reset conversation"):
        st.session_state.chat_history = []
        st.rerun()

st.markdown("<div class='neon-title'>CampusOS</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='neon-subtitle'>Ask about academics, fees, or anything else on your mind.</div>",
    unsafe_allow_html=True,
)

for role, text in st.session_state.chat_history:
    avatar = "🧑‍🎓" if role == "human" else "🤖"
    with st.chat_message("user" if role == "human" else "assistant", avatar=avatar):
        st.markdown(text)

user_query = st.chat_input("Type your question here...")

if user_query:
    st.session_state.chat_history.append(("human", user_query))
    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(user_query)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking..."):
            result = app.invoke({
                "programme": st.session_state.programme,
                "messages": [("human", user_query)]
            })
            answer = result["messages"][-1].content
        st.markdown(answer)

    st.session_state.chat_history.append(("ai", answer))