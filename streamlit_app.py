"""A separate Streamlit interface for the supplied college-assistant graph."""
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing import Annotated, TypedDict

load_dotenv()
st.set_page_config(page_title="College Assistant", page_icon="✦", layout="wide")

BASE_DIR = Path(__file__).parent


def get_api_key():
    """Read a local .env key first, then Streamlit Cloud secrets."""
    return os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")


@st.cache_resource(show_spinner="Preparing your college knowledge base…")
def build_graph(api_key: str):
    """UI-friendly version of the supplied LangGraph workflow."""
    academic_pdf = BASE_DIR / "academics_handbook.pdf"
    fee_pdf = BASE_DIR / "fee_structure.pdf"
    missing = [str(path.name) for path in (academic_pdf, fee_pdf) if not path.exists()]
    if missing:
        raise FileNotFoundError("Add " + " and ".join(missing) + " beside streamlit_app.py.")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    def build_retriever(pdf_path: Path):
        documents = PyPDFLoader(str(pdf_path)).load()
        chunks = RecursiveCharacterTextSplitter(
            chunk_size=800, chunk_overlap=100
        ).split_documents(documents)
        return FAISS.from_documents(chunks, embeddings).as_retriever(
            search_kwargs={"k": 4}
        )

    academic_retriever = build_retriever(academic_pdf)
    fee_retriever = build_retriever(fee_pdf)
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.4, api_key=api_key)

    class State(TypedDict):
        programme: str
        messages: Annotated[list, add_messages]
        query_type: str
        retrieved_context: str

    def classifier_node(state: State):
        query = state["messages"][-1].content
        prompt = (
            "Classify this student query into exactly one category: academic, fee, "
            "or general. Academic includes attendance, exams, grading, credits, "
            "promotion, course structure, summer training, and degree requirements. "
            "Fee includes tuition, payments, refunds, late charges, scholarships, and money. "
            "General covers greetings and other casual talk.\n\n"
            f"Query: {query}\nReturn only: academic, fee, or general."
        )
        answer = llm.invoke(prompt).content.strip().lower()
        category = "academic" if "academic" in answer else "fee" if "fee" in answer else "general"
        return {"query_type": category}

    def academic_rag_node(state: State):
        docs = academic_retriever.invoke(state["messages"][-1].content)
        return {"retrieved_context": "\n\n".join(doc.page_content for doc in docs)}

    def fee_rag_node(state: State):
        docs = fee_retriever.invoke(state["messages"][-1].content)
        return {"retrieved_context": "\n\n".join(doc.page_content for doc in docs)}

    def general_node(_: State):
        return {"retrieved_context": "NO_RETRIEVAL_NEEDED"}

    def response_node(state: State):
        query = state["messages"][-1].content
        programme = state.get("programme", "student")
        context = state["retrieved_context"]
        if context == "NO_RETRIEVAL_NEEDED":
            prompt = f"You are a friendly college assistant talking to a {programme} student. Answer: {query}"
        else:
            prompt = (
                f"You are a college assistant helping a {programme} student. Use only this official context. "
                "If it has programme-specific figures, highlight the relevant one. "
                "Give a clear, friendly, precise answer.\n\n"
                f"Context:\n{context}\n\nQuestion: {query}"
            )
        return {"messages": [("ai", llm.invoke(prompt).content.strip())]}

    def route_query(state: State):
        return {"academic": "academic_rag", "fee": "fee_rag"}.get(state["query_type"], "general")

    graph = StateGraph(State)
    graph.add_node("classifier", classifier_node)
    graph.add_node("academic_rag", academic_rag_node)
    graph.add_node("fee_rag", fee_rag_node)
    graph.add_node("general", general_node)
    graph.add_node("response", response_node)
    graph.add_edge(START, "classifier")
    graph.add_conditional_edges("classifier", route_query)
    graph.add_edge("academic_rag", "response")
    graph.add_edge("fee_rag", "response")
    graph.add_edge("general", "response")
    graph.add_edge("response", END)
    return graph.compile()


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: radial-gradient(circle at 8% 5%, #ffd8e7 0, transparent 29%), radial-gradient(circle at 92% 0%, #f7d9e8 0, transparent 27%), #fffafb; }
header { background: transparent !important; }
[data-testid="stSidebar"] { background: rgba(255,255,255,.58); border-right: 1px solid rgba(255,255,255,.85); backdrop-filter: blur(24px); }
.hero { padding: 3.3rem 1rem 2rem; text-align:center; }
.hero-badge { display:inline-block; padding:.45rem .8rem; background:rgba(255,255,255,.64); border:1px solid rgba(255,255,255,.9); border-radius:999px; color:#a84e74; font-size:.82rem; letter-spacing:.04em; }
.hero h1 { font-family:'Playfair Display', serif; color:#412630; font-size:clamp(2.5rem,5vw,4.5rem); margin:.7rem 0 .4rem; letter-spacing:-.045em; }
.hero p { color:#775764; font-size:1.08rem; margin:0; }
.glass { background:rgba(255,255,255,.62); border:1px solid rgba(255,255,255,.9); border-radius:28px; padding:1rem 1.2rem; box-shadow:0 12px 42px rgba(137,66,95,.10); backdrop-filter:blur(18px); }
.stChatMessage { background:rgba(255,255,255,.62); border:1px solid rgba(255,255,255,.85); border-radius:22px; padding:.4rem .7rem; box-shadow:0 8px 24px rgba(137,66,95,.06); }
[data-testid="stChatMessageAvatarUser"], [data-testid="stChatMessageAvatarAssistant"] { width:2.35rem; height:2.35rem; min-width:2.35rem; border-radius:50%; box-shadow:inset 0 1px 1px rgba(255,255,255,.72), 0 5px 14px rgba(130,58,86,.14); }
[data-testid="stChatMessageAvatarUser"] { background:linear-gradient(135deg, #ef9fbd, #d96c96); border:1px solid rgba(255,255,255,.72); }
[data-testid="stChatMessageAvatarAssistant"] { background:linear-gradient(135deg, #fff, #f8cedc); border:1px solid rgba(217,108,150,.34); }
[data-testid="stChatMessageAvatarUser"] svg, [data-testid="stChatMessageAvatarAssistant"] svg { display:none; }
.stChatInputContainer { background:rgba(255,255,255,.74); border:1px solid rgba(255,255,255,.9); border-radius:24px; box-shadow:0 12px 35px rgba(137,66,95,.12); }
.stButton button { border-radius:999px; border:0; background:#d96c96; color:#fff; font-weight:600; }
</style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.markdown("### ✦ Your space")
    programme = st.selectbox("Programme", ["BCA", "BBA", "B.Com (H)"], label_visibility="collapsed")
    st.caption("Your programme personalizes the guidance you receive.")
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.markdown("<br><div class='glass'>Ask about academics, fees, or campus life. Answers to policy questions are grounded in your uploaded college PDFs.</div>", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <span class="hero-badge">YOUR COLLEGE COMPANION</span>
  <h1>How can I help today?</h1>
  <p>Clear answers for your academic journey, in one calm space.</p>
</div>
""", unsafe_allow_html=True)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about attendance, fees, exams, or anything else…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        api_key = get_api_key()
        if not api_key:
            reply = "Add your `GROQ_API_KEY` to `.env` locally or Streamlit Cloud Secrets before starting the assistant."
            st.info(reply)
        else:
            try:
                graph = build_graph(api_key)
                with st.spinner("Looking into that…"):
                    result = graph.invoke({"programme": programme, "messages": [("human", prompt)]})
                reply = result["messages"][-1].content
                st.markdown(reply)
            except Exception as error:
                reply = f"I couldn’t start the knowledge base yet: {error}"
                st.error(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})
