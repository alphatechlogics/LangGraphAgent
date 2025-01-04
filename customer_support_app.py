import os
from dotenv import load_dotenv
from typing import Dict, TypedDict
import streamlit as st
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables.graph import MermaidDrawMethod
from IPython.display import Image

# Load environment variables from .env if it exists
load_dotenv()

# First, try to get the API key from the local environment (.env)
api_key = os.getenv('OPENAI_API_KEY')

# If the API key is not found locally, check Streamlit secrets
if not api_key:
    api_key = st.secrets.get("OPENAI_API_KEY")

# Set the OpenAI API key if found
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
else:
    raise ValueError("OpenAI API key not found. Please set it in the .env file or Streamlit secrets.")

# Define the state class to hold customer query information
class State(TypedDict):
    query: str
    category: str
    sentiment: str
    response: str

# Define node functions

def categorize(state: State) -> State:
    """Categorize the customer query into Technical, Billing, or General."""
    prompt = ChatPromptTemplate.from_template(
        "Categorize the following customer query into one of these categories: "
        "Technical, Billing, General. Query: {query}"
    )
    chain = prompt | ChatOpenAI(temperature=0)
    category = chain.invoke({"query": state["query"]}).content
    return {"category": category}

def analyze_sentiment(state: State) -> State:
    """Analyze the sentiment of the customer query as Positive, Neutral, or Negative."""
    prompt = ChatPromptTemplate.from_template(
        "Analyze the sentiment of the following customer query. "
        "Respond with either 'Positive', 'Neutral', or 'Negative'. Query: {query}"
    )
    chain = prompt | ChatOpenAI(temperature=0)
    sentiment = chain.invoke({"query": state["query"]}).content
    return {"sentiment": sentiment}

def handle_technical(state: State) -> State:
    """Provide a technical support response to the query."""
    prompt = ChatPromptTemplate.from_template(
        "Provide a technical support response to the following query: {query}"
    )
    chain = prompt | ChatOpenAI(temperature=0)
    response = chain.invoke({"query": state["query"]}).content
    return {"response": response}

def handle_billing(state: State) -> State:
    """Provide a billing support response to the query."""
    prompt = ChatPromptTemplate.from_template(
        "Provide a billing support response to the following query: {query}"
    )
    chain = prompt | ChatOpenAI(temperature=0)
    response = chain.invoke({"query": state["query"]}).content
    return {"response": response}

def handle_general(state: State) -> State:
    """Provide a general support response to the query."""
    prompt = ChatPromptTemplate.from_template(
        "Provide a general support response to the following query: {query}"
    )
    chain = prompt | ChatOpenAI(temperature=0)
    response = chain.invoke({"query": state["query"]}).content
    return {"response": response}

def escalate(state: State) -> State:
    """Escalate the query to a human agent due to negative sentiment."""
    return {"response": "This query has been escalated to a human agent due to its negative sentiment."}

def route_query(state: State) -> str:
    """Route the query based on its sentiment and category."""
    if state["sentiment"] == "Negative":
        return "escalate"
    elif state["category"] == "Technical":
        return "handle_technical"
    elif state["category"] == "Billing":
        return "handle_billing"
    else:
        return "handle_general"

# Create and configure the graph
workflow = StateGraph(State)

# Add nodes for categorization, sentiment analysis, and handling different queries
workflow.add_node("categorize", categorize)
workflow.add_node("analyze_sentiment", analyze_sentiment)
workflow.add_node("handle_technical", handle_technical)
workflow.add_node("handle_billing", handle_billing)
workflow.add_node("handle_general", handle_general)
workflow.add_node("escalate", escalate)

# Define edges for the workflow
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

# Set entry point
workflow.set_entry_point("categorize")

# Compile the workflow into an executable app
app = workflow.compile()

# Function to run customer support processing through LangGraph workflow
def run_customer_support(query: str) -> Dict[str, str]:
    """Process a customer query through the LangGraph workflow.
    
    Args:
        query (str): The customer's query
        
    Returns:
        Dict[str, str]: A dictionary containing the query's category, sentiment, and response
    """
    results = app.invoke({"query": query})
    return {
        "category": results["category"],
        "sentiment": results["sentiment"],
        "response": results["response"]
    }

# Streamlit Interface

# Title of the app
st.title("Customer Support Agent")

# Input box for customer query
query = st.text_input("Enter your query:")

if st.button("Submit Query"):
    if query:
        # Run the customer support function (invoke the LangGraph workflow)
        result = run_customer_support(query)
        # Display the results
        st.write(f"Category: {result['category']}")
        st.write(f"Sentiment: {result['sentiment']}")
        st.write(f"Response: {result['response']}")
    else:
        st.write("Please enter a query.")

# Display the workflow graph
st.subheader("Customer Support Workflow Graph")

# Visualize and display the workflow graph in Mermaid format
graph = app.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API)
st.image(graph)
