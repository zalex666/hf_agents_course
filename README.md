---
title: Template Final Assignment
emoji: 🕵🏻‍♂️
colorFrom: indigo
colorTo: indigo
sdk: gradio
sdk_version: 5.25.2
app_file: app.py
pinned: false
hf_oauth: true
# optional, default duration is 8 hours/480 minutes. Max duration is 30 days/43200 minutes.
hf_oauth_expiration_minutes: 480
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# LangGraph Agent with OpenAI

This project implements an AI agent using LangGraph and OpenAI models. The agent can answer questions, perform calculations, and search for information using various tools.

## Features

- Uses LangGraph for agent workflow management
- Integrates with OpenAI models 
- Includes various tools:
  - Basic math operations (add, subtract, multiply, divide, modulus)
  - Web search via Tavily
  - Wikipedia search
  - ArXiv paper search
  - Similar question search from vector database
- Vector store for retrieving similar questions and answers
- Gradio web interface for easy interaction

## Setup

### 1. Environment Variables

Create a `.env` file in the root directory with the following variables:

```
OPENAI_API_KEY=your_openai_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

Optionally, you can also add:
```
GOOGLE_API_KEY=your_google_api_key_here
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Application

```bash
python app.py
```

This will start the Gradio web interface where you can interact with the agent.

## Customizing the Agent

### Changing the LLM Provider

You can modify the LLM provider and model in `app.py`:

```python
agent = BasicAgent(provider="openai", model_name="gpt-4.1-2025-04-14")
```

Supported providers:
- `openai` (default): Uses OpenAI models 
- `google`: Uses Google Generative AI models

### Adding Custom Tools

You can add new tools in `agent.py` by creating functions decorated with `@tool`:

```python
@tool
def your_tool_name(param: type) -> return_type:
    """Tool description.
    
    Args:
        param: Parameter description
    """
    # Tool implementation
    return result
```

Then add your tool to the `tools` list:

```python
tools = [
    multiply, add, subtract, divide, modulus,
    wiki_search, web_search, arvix_search,
    your_tool_name,  # Add your new tool here
]
```

## How It Works

1. The agent uses LangGraph to manage the workflow between different components.
2. When a question is received, it first checks for similar questions in the vector database.
3. The LLM (OpenAI by default) processes the question and decides what tools to use.
4. If tools are needed, they are executed and the results are fed back to the LLM.
5. The process continues until the LLM provides a final answer.

## Architecture

The agent is built using a state graph with the following components:

- **Retriever Node**: Searches for similar questions in the vector database
- **Assistant Node**: Processes the question and decides what to do next
- **Tools Node**: Executes tools requested by the assistant

The graph flow is:
1. START → Retriever
2. Retriever → Assistant
3. Assistant → Tools (if needed) or END
4. Tools → Assistant

## License

This project is open source and available under the MIT License.