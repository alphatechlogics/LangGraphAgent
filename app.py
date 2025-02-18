import os
import asyncio
import glob
from dotenv import load_dotenv
from typing import Dict, TypedDict, Optional

import streamlit as st

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from browser_use import Agent, AgentHistoryList, ActionResult  # Import global config

import subprocess
from browser_use import Agent, AgentHistoryList, ActionResult  # Your other imports
from browser_use import BrowserConfig
# Attempt to install the chromium browser for Playwright if not already installed
try:
    subprocess.run(["playwright", "install", "chromium"], check=True)
except Exception as e:
    print(f"Warning: Could not install chromium via Playwright: {e}")

# ─────────────────────────────────────────────────────────────────────
# 1) Load environment (.env)
# ─────────────────────────────────────────────────────────────────────
load_dotenv()

# Unset DISPLAY to help ensure headless behavior.
os.environ.pop("DISPLAY", None)

browser_config = BrowserConfig(
    headless=True,               # Run browser without a visible UI
    disable_security=True,       # Disable security features (if needed)
    extra_chromium_args=["--no-sandbox"]  # Additional Chromium arguments
)
# ─────────────────────────────────────────────────────────────────────
# 2) ChatOpenAI initialization
# ─────────────────────────────────────────────────────────────────────


def get_llm():
    """Returns a ChatOpenAI instance using OPENAI_API_KEY from environment."""
    return ChatOpenAI(
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )


def get_llm_browser():
    """Returns a ChatOpenAI instance for the browser agent (e.g., GPT-4)."""
    return ChatOpenAI(
        model="gpt-4o",  # Adjust if needed
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

# ─────────────────────────────────────────────────────────────────────
# 3) TypedDict for state
# ─────────────────────────────────────────────────────────────────────


class State(TypedDict, total=False):
    query: str
    category: str
    sentiment: str
    response: str
    agent_history: AgentHistoryList  # Store the full agent history here

# ─────────────────────────────────────────────────────────────────────
# 4) Node-like functions
# ─────────────────────────────────────────────────────────────────────


def categorize(state: State) -> State:
    prompt = ChatPromptTemplate.from_template(
        "Categorize the following customer query into one of these categories: "
        "Technical, Billing, General. Query: {query}"
    )
    chain = prompt | get_llm()
    category = chain.invoke({"query": state["query"]}).content.strip()
    state["category"] = category
    return state


def analyze_sentiment(state: State) -> State:
    prompt = ChatPromptTemplate.from_template(
        "Analyze the sentiment of the following customer query. "
        "Respond with either 'Positive', 'Neutral', or 'Negative'. "
        "Query: {query}"
    )
    chain = prompt | get_llm()
    sentiment = chain.invoke({"query": state["query"]}).content.strip()
    state["sentiment"] = sentiment
    return state


def handle_technical(state: State) -> State:
    prompt = ChatPromptTemplate.from_template(
        "Provide a technical support response to the following query: {query}"
    )
    chain = prompt | get_llm()
    response = chain.invoke({"query": state["query"]}).content.strip()
    state["response"] = response
    return state


def handle_billing(state: State) -> State:
    prompt = ChatPromptTemplate.from_template(
        "Provide a billing support response to the following query: {query}"
    )
    chain = prompt | get_llm()
    response = chain.invoke({"query": state["query"]}).content.strip()
    state["response"] = response
    return state

# ─────────────────────────────────────────────────────────────────────
# 5) Browser Agent for general queries using a BrowserContext
# ─────────────────────────────────────────────────────────────────────


async def run_browser_agent(task: str) -> AgentHistoryList:
    # Import Browser from browser_use.
    from browser_use import Browser

    # Create a Browser instance using our BrowserConfig.
    browser = Browser(config=browser_config)

    # Create the Agent by passing the browser instance directly.
    agent = Agent(task=task, llm=get_llm_browser(), browser=browser)
    history = await agent.run()
    return history


async def handle_general(state: State) -> State:
    """
    For general queries, use a browser-based approach.
    """
    task = (
        "You are a customer support agent that consults online sources. "
        f"Provide a detailed, informed response to this customer query: {state['query']}"
    )
    history = await run_browser_agent(task)
    state["agent_history"] = history

    final_text = ""

    # Parse the final text from the agent history using attribute access
    if hasattr(history, "all_results"):
        for action in reversed(history.all_results):
            if getattr(action, "is_done", False):
                if hasattr(action, "extracted_content") and action.extracted_content:
                    final_text = action.extracted_content.strip()
                    break
                if hasattr(action, "done") and isinstance(action.done, dict):
                    text_val = action.done.get("text", "").strip()
                    if text_val:
                        final_text = text_val
                        break

    if not final_text:
        final_text = "No final content found."

    state["response"] = final_text
    return state


def escalate(state: State) -> State:
    state["response"] = "This query has been escalated to a human agent due to negative sentiment."
    return state


def route_query(state: State) -> str:
    if state["sentiment"].lower() == "negative":
        return "escalate"
    elif state["category"].lower() == "technical":
        return "handle_technical"
    elif state["category"].lower() == "billing":
        return "handle_billing"
    else:
        return "handle_general"

# ─────────────────────────────────────────────────────────────────────
# 6) Workflow
# ─────────────────────────────────────────────────────────────────────


async def run_workflow(state: State) -> State:
    state = categorize(state)
    state = analyze_sentiment(state)
    next_step = route_query(state)

    if next_step == "handle_technical":
        state = handle_technical(state)
    elif next_step == "handle_billing":
        state = handle_billing(state)
    elif next_step == "handle_general":
        state = await handle_general(state)
    else:
        state = escalate(state)

    return state

# ─────────────────────────────────────────────────────────────────────
# 7) Main orchestrator for user queries
# ─────────────────────────────────────────────────────────────────────


async def process_query(query: str, api_key: str):
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    elif not os.getenv("OPENAI_API_KEY"):
        return ("Error: Please provide an OpenAI API key.", None)

    try:
        init_state: State = {
            "query": query,
            "category": "",
            "sentiment": "",
            "response": ""
        }
        final_state = await run_workflow(init_state)
        final_text = final_state["response"]
        agent_history = final_state.get("agent_history", None)
        return (final_text, agent_history)
    except Exception as e:
        return (f"Error: {str(e)}", None)

# ─────────────────────────────────────────────────────────────────────
# 8) Streamlit UI
# ─────────────────────────────────────────────────────────────────────


def main():
    st.title("Customer Support Agent with Browser Use (Streamlit)")
    st.write(
        "This agent categorizes customer queries and uses a browser-based "
        "agent to provide informed answers when the query is general."
    )

    api_key = st.text_input(
        "OpenAI API Key", type="password", placeholder="sk-...")
    query = st.text_input("Customer Query", "Who is Elon Musk?")

    if st.button("Submit Query"):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            final_text, agent_history = loop.run_until_complete(
                process_query(query, api_key))

            st.subheader("Agent Response (Final Answer)")
            st.write(final_text)

            gif_files = glob.glob("agent_history.gif")
            if gif_files:
                st.subheader("Browser Agent GIF")
                st.image(gif_files[0], caption="Agent History")

            if agent_history:
                st.subheader("Complete Agent Response")
                st.write(str(agent_history))

        finally:
            loop.close()


if __name__ == "__main__":
    main()
