"""
title: ...
author: ...
date: 2025-1-27
version: 0.1
description: ...
"""
# Moving the following line into the comments above will pip install all the requirements on 
# the Open WebUI Pipeline server. Moving the requirements out of the comments keeps them from
# being installed on every neww installation of the pipeline code.
# requirements: typing_extensions, pydantic==2.7.4, langchain, langchain-ollama, langchain-core, langchain-community, langgraph

import logging

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_ollama import ChatOllama
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel
from rich.logging import RichHandler
import traceback
from typing_extensions import List, TypedDict, Annotated
from typing import (Sequence)


# The Pipeline Class is used for running this code using Open Webui Pipeline project
# https://github.com/open-webui/pipelines
class Pipeline:
    # Valves are used to define the pipeline parameters. The valve parameters are 
    # configurable in the Open WebUI front end
    class Valves(BaseModel):
        OLLAMA_API_BASE_URL: str = "http://open-webui-ollama.ai-agent-0-1.svc.cluster.local:11434"
        OLLAMA_API_KEY: str = "if you are hosting ollama, put api key here"
        LOG_LEVEL: int = 10 # 10 is DEBUG, 20 is INFO, 30 is WARNING, 40 is ERROR, 50 is CRITICAL

    def __init__(self):
        self.name = "AI Agent 0.4.0"
        self.valves = self.Valves()

        logging.basicConfig(
            level=self.valves.LOG_LEVEL,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler()]
        )

        self.logger = logging.getLogger(self.name)

        # # Agent LLM for user conversations AND tool calling
        # # NOTE: we could use different LLMs for conversations and tools, but that would
        # #       require additional GPU resources.  
        self.ollama_conversation_model = "mistral-nemo:latest" # general conversation model
        self.ollama_conversation_temperature = 0 # set to 0 for predictable results
        self.ollama_conversation_num_ctx = 96000 # max value is 128000 for mistral-nemo
        self.ollama_converstation_num_predict = 4096

        self.llm = ChatOllama(
            model=self.ollama_conversation_model, 
            base_url=self.valves.OLLAMA_API_BASE_URL,
            temperature=self.ollama_conversation_temperature,
            num_ctx=self.ollama_conversation_num_ctx,
            num_predict=self.ollama_converstation_num_predict)

    def pipe(self, 
             user_message: str, # contains the current user query sent from front-end
             model_id: str, 
             messages: List[dict], # contains the history of the chat conversation sent from front-end
             body: dict): # contains a dict about request sent from front-end including user account information
        self.logger.info(f"Pipeline Invoked: {self.name}: {model_id}")
        self.logger.debug(f"Pipeline Request Body: {body}")

        graph_builder = StateGraph(AgentState)
        self.logger.debug(f"Building Graph...")
        query_agent = QueryAgent(logger=self.logger, llm=self.llm)
        graph_builder.add_node("query_agent", query_agent.exec)
        graph_builder.add_edge(START, "query_agent")
        graph_builder.add_edge("query_agent", END)

        try:
            graph = graph_builder.compile()
            graph_nodes_allowed_to_respond = ["query_agent"]
            for response_message, metadata in graph.stream(
                                                {"query": user_message,
                                                 "messages": Util().convert_openwebui_messages_to_langchain(messages),
                                                 "documents": [],
                                                 "eval_count": 0}, 
                                                 stream_mode="messages",
                                                 config={}
                                                ):
                if metadata["langgraph_node"] in graph_nodes_allowed_to_respond:
                    yield response_message.content                

        except Exception as e:
            self.logger.error(f"An error occurred: {str(e)}")
            self.logger.error(f"{traceback.format_exc()}")
            raise


class AgentState(TypedDict):
    query: str
    messages: Annotated[Sequence[BaseMessage], add_messages]
    documents: List[Document]
    eval_count: int


class QueryAgent:
    def __init__(self, logger, llm):
        self.logger = logger
        self.llm = llm
        self.agent_prompt = "You are a helpful assistant. Answer the user's question."

    def exec(self, state: AgentState):
        try:
            # Call the LLM to either respond or make a tool call
            prompt = ChatPromptTemplate.from_messages([
                ("system", self.agent_prompt), 
                ("placeholder", "{conversation}")
                ])

            ai_query = prompt | self.llm

            response = ai_query.invoke({"conversation": state["messages"]})

        except Exception as e:
            self.logger.error(e)
            response = "I'm sorry, I seem to be having trouble with that. Try opening a new chat and asking me the same request, or try rephrasing the request."

        self.logger.debug(f"Query Agent: LLM Response: {response}")
        return {"messages": [response]}

class Util:
    def __init__(self):
        pass

    def convert_openwebui_messages_to_langchain(self, messages):
        converted_messages = []
        for message in messages:
            if message["role"] == "user":
                converted_messages.append(HumanMessage(content=message["content"]))
            else:
                converted_messages.append(AIMessage(content=message["content"]))
        return converted_messages
