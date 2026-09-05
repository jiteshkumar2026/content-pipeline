import os
from typing import TypedDict
from dotenv import load_dotenv
load_dotenv()

# creating a state 
class pipelinestate(TypedDict):
    raw_input:str
    edited_text:str
    script_text:str
    final_output:str

#use LLMs
from langchain_groq import ChatGroq
llm=ChatGroq(model="openai/gpt-oss-120b", temperature=0.7)

# create nodes 
# Node 1

def editor_node(state:pipelinestate)-> dict:
    """Stage 1: Cleans up grammar, removes typos and refines the tone."""
    prompt=(
       """you are an expert copyeditor.clean up the following raw text. """
       """fix any grammatical errors, spelling mistakes, and smooth out the transition flow"""
       """while keeping the core message intact. Return only the edited text.\n\n"""
       f"Text:\n{state['raw_input']}"
   )
    response=llm.invoke(prompt)
    return {"edited_text":response.content.strip()}

# Node 2

def scriptwriter_node(state:pipelinestate)->dict:
    """stage 2: formats the clean text into an engaging video script style."""
    print("\n--[stage2] Executing scriptwriter node--")

    prompt=(
        """You are a charismatic Youtube content creator .Take this edited text and transform"""
        """ it into a highly engaging ,punchy ,conversational video script hook."""
        """make it sound like a real person speaking passionately. Return only the script content.\n\n"""
        f"Edited text:\n{state['edited_text']}"
    )
    response=llm.invoke(prompt)
    return {"script_text":response.content.strip()}

# Node 3

def tranlator_node(state:pipelinestate)->dict:
    """stage 3: Translates the script into natural flowing Hinglish"""
    print("\n---[Stage 3]Executing Hinglish Translator Node---")

    prompt = (
        "You are an expert content localizer for the Indian market. Take the following script "
        "and convert it into natural, flowing 'Hinglish'. Do not simply translate it sentence-by-sentence "
        "or repeat information. Alternating comfortably between Hindi and English phrases just like "
        "an intellectual tech educator would speak naturally on a live stream. Keep the energy high! "
        "Return only the final Hinglish text.\n\n"
        f"Script:\n{state['script_text']}"
    )

    response=llm.invoke(prompt)
    return {"final_output":response.content.strip()}


#connecting these nodes using edges

from langgraph.graph import StateGraph,START,END

#create the graph
graph=StateGraph(pipelinestate)

#adding nodes in the graph
graph.add_node("editor",editor_node)
graph.add_node("scriptwriter",scriptwriter_node)
graph.add_node("translator",tranlator_node)


#add edges (sequential->one after another)

graph.add_edge(START,"editor")
graph.add_edge("editor","scriptwriter")
graph.add_edge("scriptwriter","translator")
graph.add_edge("translator",END)


#compile ther graph
app=graph.compile()
result=app.invoke({
    "raw_input":"AI agents are the future of tech. They can think, plan, and act on their own. LangGraph helps you build these agents with proper control and memory."
})

#prinitng result
print("\n\n--HERE is the Result--")
print(result['final_output'])