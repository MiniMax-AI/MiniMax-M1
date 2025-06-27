# Guia de Uso de Function Call no MiniMax-M1

[FunctionCall中文使用指南](./function_call_guide_cn.md)

## 📖 Introdução

O modelo MiniMax-M1 possui suporte para chamadas de funções (Function Call), permitindo que o modelo identifique quando funções externas precisam ser chamadas e gere os parâmetros dessas chamadas em um formato estruturado. Este documento fornece instruções detalhadas sobre como utilizar o recurso de chamadas de funções do MiniMax-M1.

## 🚀 Início Rápido

### Usando o Template de Chat

O MiniMax-M1 utiliza um template específico de chat para lidar com chamadas de funções. Este template é definido no arquivo `tokenizer_config.json` e pode ser utilizado no seu código através do template.

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

# Modelo de carga e tokenizador
tokenizer = AutoTokenizer.from_pretrained(model_id)
prompt = "What's the weather like in Shanghai today?"
messages = [
    {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant created by Minimax based on MiniMax-M1 model."}]},
    {"role": "user", "content": [{"type": "text", "text": prompt}]},
]

# Habilitar ferramentas de chamada de função
tools = get_default_tools()

# Aplicar modelo de bate-papo e adicionar definições de ferramentas
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    tools=tools
)

# Enviar solicitação
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

## 🛠️ Definição de Function Call

### Estrutura da Função

As funções precisam ser definidas no campo `tools` do corpo da requisição. Cada função é composta pelos seguintes elementos:

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

**Descrição dos Campos:**

* `name`: Nome da função
* `description`: Descrição da função
* `parameters`: Definição dos parâmetros da função

  * `properties`: Definições dos parâmetros, onde a chave é o nome do parâmetro e o valor contém a descrição
  * `required`: Lista de parâmetros obrigatórios
  * `type`: Tipo de dado (geralmente "object")

### Formato Interno de Processamento do Modelo

Internamente, as definições de funções são convertidas para um formato especial e concatenadas ao texto de entrada:

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

### Formato de Saída do Modelo

O modelo gera chamadas de função no seguinte formato:

```xml
<think>
Ok, vou procurar a versão mais recente do OpenAI e do Gemini.
</think>
<tool_calls>
{"name": "search_web", "arguments": {"query_tag": ["technology", "events"], "query_list": ["\"OpenAI\" \"latest\" \"release\""]}}
{"name": "search_web", "arguments": {"query_tag": ["technology", "events"], "query_list": ["\"Gemini\" \"latest\" \"release\""]}}
</tool_calls>
```

## 📥 Processamento dos Resultados da Function Call

### Fazendo o Parse das Chamadas de Função

Você pode utilizar o código abaixo para extrair as chamadas de função a partir da saída do modelo:

```python
import re
import json

def parse_function_calls(content: str):
    """
    Parse function calls from model output
    """
    function_calls = []
    
    # Corresponder conteúdo dentro das tags <tool_calls>
    tool_calls_pattern = r"<tool_calls>(.*?)</tool_calls>"
    tool_calls_match = re.search(tool_calls_pattern, content, re.DOTALL)
    
    if not tool_calls_match:
        return function_calls
    
    tool_calls_content = tool_calls_match.group(1).strip()
    
    # Analisar cada chamada de função (um objeto JSON por linha)
    for line in tool_calls_content.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        try:
            # Chamada de função de formato JSON de análise
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

# Exemplo: Manipular função de consulta de clima
def execute_function_call(function_name: str, arguments: dict):
    """
    Execute function call and return result
    """
    if function_name == "get_current_weather":
        location = arguments.get("location", "Unknown location")
        # Resultado da execução da função de construção
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
        # Simular resultados de pesquisa
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

### Retornando os Resultados das Funções para o Modelo

Após interpretar e executar as funções, você deve adicionar os resultados na sequência de mensagens, para que o modelo os utilize nas respostas seguintes.

#### Resultado Único

Se o modelo solicitar a função `search_web`, retorne no seguinte formato, com o campo `name` igual ao nome da ferramenta:

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

Formato correspondente no input do modelo:

```
<beginning_of_sentence>tool name=tools
tool name: search_web
tool result: test_result
<end_of_sentence>
```

#### Vários Resultados

Se o modelo solicitar simultaneamente `search_web` e `get_current_weather`, envie da seguinte forma, use o campo `content` para conter vários resultados:

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

Formato correspondente no input do modelo:
```
<beginning_of_sentence>tool name=tools
Tool name: search_web
Tool result: resultado1

Tool name: get_current_weather
Tool result: resultado2<end_of_sentence>
```

Embora esse seja o formato recomendado, desde que a entrada seja clara para o modelo, os valores de `name` e `content` podem ser adaptados conforme a necessidade.
