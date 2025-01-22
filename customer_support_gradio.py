import os
import asyncio
from dotenv import load_dotenv
from typing import Dict, TypedDict

import gradio as gr

from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from browser_use import Agent

# Load environment variables (including OPENAI_API_KEY) from .env
load_dotenv()

# Define a TypedDict to hold state information.
class State(TypedDict):
    query: str
    category: str
    sentiment: str
    response: str

# Initialize our language models.
# We use llm_standard for normal tasks and llm_browser for browser-based tasks.
llm_standard = ChatOpenAI(temperature=0)
llm_browser = ChatOpenAI(model="gpt-4o", temperature=0)

# Node functions for our workflow.
def categorize(state: State) -> State:
    prompt = ChatPromptTemplate.from_template(
        "Categorize the following customer query into one of these categories: "
        "Technical, Billing, General. Query: {query}"
    )
    chain = prompt | llm_standard
    category = chain.invoke({"query": state["query"]}).content.strip()
    return {"category": category}

def analyze_sentiment(state: State) -> State:
    prompt = ChatPromptTemplate.from_template(
        "Analyze the sentiment of the following customer query. "
        "Respond with either 'Positive', 'Neutral', or 'Negative'. Query: {query}"
    )
    chain = prompt | llm_standard
    sentiment = chain.invoke({"query": state["query"]}).content.strip()
    return {"sentiment": sentiment}

def handle_technical(state: State) -> State:
    prompt = ChatPromptTemplate.from_template(
        "Provide a technical support response to the following query: {query}"
    )
    chain = prompt | llm_standard
    response = chain.invoke({"query": state["query"]}).content.strip()
    return {"response": response}

def handle_billing(state: State) -> State:
    prompt = ChatPromptTemplate.from_template(
        "Provide a billing support response to the following query: {query}"
    )
    chain = prompt | llm_standard
    response = chain.invoke({"query": state["query"]}).content.strip()
    return {"response": response}

async def run_browser_agent(task: str) -> str:
    # Run the browser-use agent asynchronously.
    agent = Agent(task=task, llm=llm_browser)
    result = await agent.run()
    return result

def handle_general(state: State) -> State:
    """
    For general queries, we use the browser agent to consult online resources.
    We call the async function with asyncio.run and then extract only the final answer.
    """
    task = (
        "You are a customer support agent that consults online sources. "
        f"Provide a detailed, informed response to this customer query: {state['query']}"
    )
    result = asyncio.run(run_browser_agent(task))
    final_text = ""
    
    if isinstance(result, str):
        final_text = result.strip()
    elif hasattr(result, "all_results"):
        # Iterate over the list of ActionResults to extract the final done answer
        for action in result.all_results:
            # Check if the action is marked as done and has extracted content
            if action.get("is_done") and action.get("extracted_content"):
                final_text = action.get("extracted_content").strip()
        # Fallback in case no done action is found
        if not final_text:
            final_text = str(result).strip()
    else:
        final_text = str(result).strip()
    
    return {"response": final_text}

def escalate(state: State) -> State:
    return {"response": "This query has been escalated to a human agent due to negative sentiment."}

def route_query(state: State) -> str:
    """Determine which node to route to based on sentiment and category."""
    if state["sentiment"].lower() == "negative":
        return "escalate"
    elif state["category"].lower() == "technical":
        return "handle_technical"
    elif state["category"].lower() == "billing":
        return "handle_billing"
    else:
        return "handle_general"

# Create the workflow graph.
workflow = StateGraph(State)
workflow.add_node("categorize", categorize)
workflow.add_node("analyze_sentiment", analyze_sentiment)
workflow.add_node("handle_technical", handle_technical)
workflow.add_node("handle_billing", handle_billing)
workflow.add_node("handle_general", handle_general)
workflow.add_node("escalate", escalate)

workflow.add_edge("categorize", "analyze_sentiment")
workflow.add_conditional_edges(
    "analyze_sentiment",
    route_query,
    {
        "handle_technical": "handle_technical",
        "handle_billing": "handle_billing",
        "handle_general": "handle_general",
        "escalate": "escalate"
    }
)
workflow.add_edge("handle_technical", END)
workflow.add_edge("handle_billing", END)
workflow.add_edge("handle_general", END)
workflow.add_edge("escalate", END)
workflow.set_entry_point("categorize")
app = workflow.compile()

def run_customer_support(query: str, api_key: str) -> str:
    """
    Process the customer query through the workflow.
    Use the provided API key, or if none is given, fall back to the .env value.
    Only the final answer is returned.
    """
    # If no API key is provided in the UI, try to read it from the environment.
    if not api_key.strip():
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            return "Please provide a valid OpenAI API key."
    os.environ["OPENAI_API_KEY"] = api_key
    results = app.invoke({"query": query})
    # Return only the final answer (the response part)
    return results.get("response", "No response generated.")

# Build the Gradio UI.
with gr.Blocks(title="Customer Support Agent with Browser Use") as demo:
    gr.Markdown("# Customer Support Agent with Browser Use")
    gr.Markdown("This agent categorizes customer queries and uses a browser-based agent to provide informed answers.")
    
    with gr.Row():
        with gr.Column():
            # The API key textbox (if left empty, the app will try to use the .env key)
            api_key_input = gr.Textbox(label="OpenAI API Key", type="password", placeholder="sk-...", value="")
            query_input = gr.Textbox(label="Customer Query", placeholder="Enter your query here...", lines=3)
            submit_btn = gr.Button("Submit Query")
        with gr.Column():
            output_box = gr.Textbox(label="Agent Response", lines=10, interactive=False)
    
    submit_btn.click(fn=run_customer_support, inputs=[query_input, api_key_input], outputs=output_box)

# Launch the Gradio interface.
demo.launch(share=True)
