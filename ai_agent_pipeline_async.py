"""
title: Pipeline to connect to a service running LangGraph Agent
author: Paul Jensen
date: 2025-1-27
version: 0.4
description: This is a pipeline to connect to a service running LangGraph Agent. The pipeline is loaded by the Open WebUI Pipeline server. Deploy the service using the Open WebUI Admin Panel.
"""
#requirements: pydantic, requests

import requests
import json
from typing import List, Union, Generator, Iterator
# try:
#     from pydantic.v1 import BaseModel
# except Exception:
#     from pydantic import BaseModel
from pydantic import BaseModel


class Pipeline:

    class Valves(BaseModel):
        AI_AGENT_API_BASE_URL: str = "<URL to AI Agent API>/openwebui-pipelines/api/stream"

    def __init__(self):
        self.id = "AI Agent"
        self.name = "AI Agent"
        self.valves = self.Valves()

    async def on_startup(self):
        print(f"on_startup: {__name__}")
        pass

    async def on_shutdown(self):
        print(f"on_shutdown: {__name__}")
        pass

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
            ) -> Union[str, Generator, Iterator]:

        headers = {
            'accept': 'application/json',
            'Content-Type': 'application/json'
        }

        data = {
            "user_message": user_message,
            "model_id": model_id,
            "messages": [[msg['role'], msg['content']] for msg  in messages],
            "body": body
            }
        
        response = requests.post(self.valves.AI_AGENT_API_BASE_URL, json=data, headers=headers, stream=True)
        
        response.raise_for_status()
        
        return response.iter_content(chunk_size=1024)
