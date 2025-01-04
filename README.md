# 🧑‍💻 Intelligent Customer Support Agent with LangGraph 🤖

## 📚 Project Overview

This repository presents an intelligent customer support agent powered by LangGraph, a robust tool designed for building intricate workflows with language models. The agent is programmed to classify customer queries, evaluate sentiment, and deliver suitable responses or escalate inquiries when necessary.

## 💡 Purpose

In today's competitive business world, delivering efficient and accurate customer support is essential. By automating initial customer interactions, response times can be significantly shortened, enhancing customer satisfaction. This project demonstrates how advanced language models combined with graph-based workflows can be used to develop an effective and scalable support system, capable of addressing a variety of customer concerns.

## 🛠 Core Components

- **State Management**: Utilizes `TypedDict` to track and manage the state of each customer interaction. 🗃️
- **Query Categorization**: Classifies queries into categories such as Technical, Billing, or General. 🏷️
- **Sentiment Analysis**: Assesses the emotional tone of customer queries. 😊😐😡
- **Response Generation**: Constructs responses based on the query's category and sentiment. 💬
- **Escalation Protocol**: Automatically escalates queries with negative sentiment to a human agent. 🚨
- **Workflow Graph**: Leverages LangGraph to build an adaptable and extendable workflow. 🔄

## 📝 How It Works

1. **Setup**: Install the necessary libraries and prepare the environment. 📦
2. **State Design**: Create a structure to store query data, including category, sentiment, and response. 🏗️
3. **Node Functions**: Define functions for categorization, sentiment analysis, and generating responses. 🔄
4. **Graph Design**: Use `StateGraph` to establish the workflow, incorporating nodes and edges to map the support process. 🌐
5. **Routing Logic**: Implement conditional routing based on query category and sentiment. 🔀
6. **Graph Compilation**: Compile the workflow into a functioning application. ⚙️
7. **Execution**: Process incoming customer queries through the workflow and generate results. 🏃‍♂️

## 📊 Workflow Diagram

The following diagram illustrates the workflow of the customer support agent. It shows the steps from query categorization and sentiment analysis to routing the query for appropriate action or escalation.

![Customer Support Workflow](image.jpg)

## ✅ Project Summary

This project highlights the versatility and power of LangGraph in creating AI-driven workflows. By combining natural language processing with a graph-based workflow approach, we've built a customer support agent capable of efficiently handling diverse queries. This system can be easily extended and customized to meet the unique needs of various businesses, and can potentially integrate with existing customer support tools and databases for enhanced functionality.

The approach demonstrated here has applications beyond customer support, showcasing how language models can be orchestrated to solve complex, multi-step challenges in various fields.

---

## 🚀 Getting Started

Follow the steps below to set up the project and run the customer support agent application.

### 1. Clone the Repository

Start by cloning this repository to your local machine:

```bash
git clone https://github.com/alphatechlogics/LangGraphAgent.git
cd LangGraphAgent
```

### 2. Create a Virtual Environment

Navigate to the project directory and create a virtual environment to manage the project's dependencies:

- For Windows:

```bash
python -m venv venv
```

- For macOS/Linux:

```bash
python3 -m venv venv
```

### 3. Activate the Virtual Environment

- On Windows:

```bash
.\venv\Scripts\activate
```

- On macOS/Linux:

```bash
source venv/bin/activate
```

### 4. Install Required Dependencies

Once the virtual environment is activated, install the necessary dependencies using pip:

```bash
pip install -r requirements.txt
```

### 5. Set Up Your OpenAI API Key

Create a .env file in the project root directory and add your OpenAI API key. This is required to interact with the OpenAI models:

```makefile
OPENAI_API_KEY=your_openai_api_key_here
```

### 6. Run the Application

After installing the dependencies and setting up the .env file, you can run the application:

```bash
streamlit run customer_support_app.py
```

### 7. Access the App

Once the app is running, open your browser and go to the following URL to interact with the customer support agent:

```bash
http://localhost:8501
```
