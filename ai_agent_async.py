import os
import json
from typing import Annotated, AsyncGenerator
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_ollama import ChatOllama
from dotenv import load_dotenv

# load environment variables from .env
load_dotenv()
# Get the LLM API key is needed
llm_api_base_url = os.getenv("LLM_API_BASE_URL", "<url for ollama service>")
api_key = os.getenv("LLM_API_KEY", None) # not used currently
llm_conversation_model = os.getenv("LLM_MODEL", "mistral-nemo:latest") # general conversation model
llm_conversation_temperature = os.getenv("LLM_TEMPERATURE", 0) # set to 0 for predictable results
llm_conversation_num_ctx = os.getenv("LLM_NUM_CTX", 96000) # max value is 128000 for mistral-nemo
llm_converstation_num_predict = os.getenv("LLM_NUM_PREDICT", 4096)

class State(TypedDict):
    messages: Annotated[list, add_messages]


llm = ChatOllama(
    model=llm_conversation_model, 
    base_url=llm_api_base_url,
    temperature=llm_conversation_temperature,
    num_ctx=llm_conversation_num_ctx,
    num_predict=llm_converstation_num_predict)

        
def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()
