"""
This is the main entry point for the agent.
It defines the workflow graph, state, tools, nodes and edges.
"""

import os
from dotenv import load_dotenv
from typing import Any, List
from typing_extensions import Literal
import google.generativeai as genai # type: ignore
from langchain_core.messages import SystemMessage, BaseMessage # type: ignore
from langchain_core.runnables import RunnableConfig # type: ignore
from langchain.tools import tool # type: ignore
from langgraph.graph import StateGraph, END # type: ignore
from langgraph.types import Command # type: ignore
from langgraph.graph import MessagesState # type: ignore
from langgraph.prebuilt import ToolNode # type: ignore


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")


class AgentState(MessagesState):
    """
    Here we define the state of the agent

    In this instance, we're inheriting from CopilotKitState, which will bring in
    the CopilotKitState fields. We're also adding a custom field, `language`,
    which will be used to set the language of the agent.
    """
    proverbs: List[str] = []
    tools: List[Any]
    # your_custom_agent_state: str = ""


@tool
def get_weather(location: str):
    """
    Get the weather for a given location.
    """
    return f"The weather for {location} is 70 degrees."


# Backend tools list
backend_tools = [
    get_weather
]

# Extract tool names from backend_tools for comparison
backend_tool_names = [tool.name for tool in backend_tools]


async def chat_node(state: AgentState, config: RunnableConfig) -> Command[Literal["tool_node", "__end__"]]:
    """
    Standard chat node adapted for Gemini.
    """

    # 1. Define the system prompt
    system_message = f"You are a helpful assistant. The current proverbs are {state.get('proverbs', [])}."

    # 2. Collect user messages
    user_messages = [msg.content for msg in state["messages"]]

    # 3. Send to Gemini
    prompt = system_message + "\n\n" + "\n".join(user_messages)
    response = model.generate_content(prompt)

    # 4. Wrap Gemini response into BaseMessage-like dict
    class GeminiMessage:
        def __init__(self, text):
            self.content = text
            self.tool_calls = []  # Gemini doesn’t have native tool calls

    response_message = GeminiMessage(response.text)

    # No tool routing (Gemini doesn’t handle tool calls natively here)
    return Command(
        goto=END,
        update={
            "messages": [response_message],
        }
    )


def route_to_tool_node(response: BaseMessage):
    """
    Stub for Gemini - currently no tool call detection.
    """
    tool_calls = getattr(response, "tool_calls", None)
    if not tool_calls:
        return False

    for tool_call in tool_calls:
        if tool_call.get("name") in backend_tool_names:
            return True
    return False


# Define the workflow graph
workflow = StateGraph(AgentState)
workflow.add_node("chat_node", chat_node)
workflow.add_node("tool_node", ToolNode(tools=backend_tools))
workflow.add_edge("tool_node", "chat_node")
workflow.set_entry_point("chat_node")

graph = workflow.compile()
