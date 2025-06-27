# MiniMax-M1 Function Call Guide

[FunctionCall中文使用指南](./function_call_guide_cn.md)

## 📖 Introduction

The MiniMax-M1 model supports function calling capabilities, enabling the model to identify when external functions need to be called and output function call parameters in a structured format. This document provides detailed instructions on how to use the function calling feature of MiniMax-M1.

## 🚀 Quick Start

### Using Chat Template

MiniMax-M1 uses a specific chat template format to handle function calls. The chat template is defined in `tokenizer_config.json`, and you can use it in your code through the template.

```python
from transformers import AutoTokenizer

def get_default_tools():
    return [
        {
          "name": "get_current_weather",
          "description": "Get the latest weather for a location",
          "parameters": {
              "type": "object", 
              "properties": {
                  "location": {
                      "type": "string", 
                      "description": "A certain city, such as Beijing, Shanghai"
                  }
              }, 
          }
          "required": ["location"],
          "type": "object"
        }
    ]

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
prompt = "What's the weather like in Shanghai today?"
messages = [
    {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant created by Minimax based on MiniMax-M1 model."}]},
    {"role": "user", "content": [{"type": "text", "text": prompt}]},
]

# Enable function call tools
tools = get_default_tools()

# Apply chat template and add tool definitions
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    tools=tools
)

# Post request
import requests
payload = {
    "model": "MiniMaxAI/MiniMax-M1-40k",
    "prompt": text,
    "max_tokens": 4000
}

response = requests.post(
                          "http://localhost:8000/v1/completions",
                          headers={"Content-Type": "application/json"},
                          json=payload,
                          stream=False,
                        )
print(response.json()["choices"][0]["text"])
```

## 🛠️ Function Call Definition

### Function Structure

Function calls need to be defined in the `tools` field of the request body. Each function consists of the following components:

```json
{
  "tools": [
    {
      "name": "search_web",
      "description": "Search function.",
      "parameters": {
        "properties": {
          "query_list": {
            "description": "Keywords for search, with list element count of 1.",
            "items": { "type": "string" },
            "type": "array"
          },
          "query_tag": {
            "description": "Classification of the query",
            "items": { "type": "string" },
            "type": "array"
          }
        },
        "required": [ "query_list", "query_tag" ],
        "type": "object"
      }
    }
  ]
}
```

**Field Descriptions:**
- `name`: Function name
- `description`: Function description
- `parameters`: Function parameter definition
  - `properties`: Parameter property definitions, where key is the parameter name and value contains detailed parameter description
  - `required`: List of required parameters
  - `type`: Parameter type (usually "object")

### Internal Model Processing Format

When processed internally by the model, function definitions are converted to a special format and concatenated to the input text:

```
<begin_of_document><beginning_of_sentence>system ai_setting=MiniMax AI
MiniMax AI is an AI assistant independently developed by MiniMax. <end_of_sentence>
<beginning_of_sentence>system tool_setting=tools
You are provided with these tools:
<tools>
{"name": "search_web", "description": "Search function.", "parameters": {"properties": {"query_list": {"description": "Keywords for search, with list element count of 1.", "items": {"type": "string"}, "type": "array"}, "query_tag": {"description": "Classification of the query", "items": {"type": "string"}, "type": "array"}}, "required": ["query_list", "query_tag"], "type": "object"}}
</tools>

If you need to call tools, please respond with <tool_calls></tool_calls> XML tags, and provide tool-name and json-object of arguments, following the format below:
<tool_calls>
{"name": <tool-name>, "arguments": <args-json-object>}
...
</tool_calls><end_of_sentence>
<beginning_of_sentence>user name=User
When were the most recent launch events for OpenAI and Gemini?<end_of_sentence>
<beginning_of_sentence>ai name=MiniMax AI
```

### Model Output Format

The model outputs function calls in the following format:

```xml
<think>
Okay, I will search for the OpenAI and Gemini latest release.
</think>
<tool_calls>
{"name": "search_web", "arguments": {"query_tag": ["technology", "events"], "query_list": ["\"OpenAI\" \"latest\" \"release\""]}}
{"name": "search_web", "arguments": {"query_tag": ["technology", "events"], "query_list": ["\"Gemini\" \"latest\" \"release\""]}}
</tool_calls>
```

## 📥 Function Call Result Processing

### Parsing Function Calls

You can use the following code to parse function calls from the model output:

```python
import re
import json

def parse_function_calls(content: str):
    """
    Parse function calls from model output
    """
    function_calls = []
    
    # Match content within <tool_calls> tags
    tool_calls_pattern = r"<tool_calls>(.*?)</tool_calls>"
    tool_calls_match = re.search(tool_calls_pattern, content, re.DOTALL)
    
    if not tool_calls_match:
        return function_calls
    
    tool_calls_content = tool_calls_match.group(1).strip()
    
    # Parse each function call (one JSON object per line)
    for line in tool_calls_content.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        try:
            # Parse JSON format function call
            call_data = json.loads(line)
            function_name = call_data.get("name")
            arguments = call_data.get("arguments", {})
            
            function_calls.append({
                "name": function_name,
                "arguments": arguments
            })
            
            print(f"Function call: {function_name}, Arguments: {arguments}")
            
        except json.JSONDecodeError as e:
            print(f"Parameter parsing failed: {line}, Error: {e}")
    
    return function_calls

# Example: Handle weather query function
def execute_function_call(function_name: str, arguments: dict):
    """
    Execute function call and return result
    """
    if function_name == "get_current_weather":
        location = arguments.get("location", "Unknown location")
        # Build function execution result
        return {
            "role": "tool", 
            "content": [
              {
                "name": function_name,
                "type": "text",
                "text": json.dumps({
                    "location": location, 
                    "temperature": "25", 
                    "unit": "celsius", 
                    "weather": "Sunny"
                }, ensure_ascii=False)
              }
            ] 
          }
    elif function_name == "search_web":
        query_list = arguments.get("query_list", [])
        query_tag = arguments.get("query_tag", [])
        # Simulate search results
        return {
            "role": "tool",
            "content": [
              {
                "name": function_name,
                "type": "text",
                "text": f"Search keywords: {query_list}, Categories: {query_tag}\nSearch results: Relevant information found"
              }
            ]
          }
    
    return None
```

### Returning Function Execution Results to the Model

After successfully parsing function calls, you should add the function execution results to the conversation history so that the model can access and utilize this information in subsequent interactions.

#### Single Result

If the model decides to call `search_web`, we suggest you to return the function result in the following format, with the `name` field set to the specific tool name.

```json
{
  "role": "tool", 
  "content": [
    {
      "name": "search_web",
      "type": "text",
      "text": "test_result"
    }
  ]
}
```

Corresponding model input format:
```
<beginning_of_sentence>tool name=tools
tool name: search_web
tool result: test_result
<end_of_sentence>
```


#### Multiple Result
If the model decides to call `search_web` and `get_current_weather` at the same time, we suggest you to return the multiple function results in the following format, use the `content` field to contain multiple results.


```json
{
  "role": "tool", 
  "content": [
    {
      "name": "search_web",
      "type": "text",
      "text": "test_result1"
    },
    {
      "name": "get_current_weather",
      "type": "text",
      "text": "test_result2"
    }
  ]
}
```

Corresponding model input format:
```
<beginning_of_sentence>tool name=tools
tool name: search_web
tool result: test_result1

tool name: get_current_weather
tool result: test_result2<end_of_sentence>
```

While we suggest following the above formats, as long as the model input is easy to understand, the specific values of `name` and `content` is entirely up to the caller.
