OpenAIModels = {
    "gpt-4-1106-preview": 128000,
    "gpt-4-vision-preview": 128000,
    "gpt-4": 8192,
    "gpt-4-0314": 8192,
    "gpt-4-0613": 8192,
    "gpt-4-32k": 32768,
    "gpt-4-32k-0314": 32768,
    "gpt-4-32k-0613": 32768,
    "gpt-3.5-turbo-1106": 4097,
    "gpt-3.5-turbo": 4097,
    "gpt-3.5-turbo-16k": 16385,
    "gpt-3.5-turbo-0301": 4097,
    "gpt-3.5-turbo-0613": 4097,
    "gpt-3.5-turbo-16k-0613": 16385,
}
"""
    List of OpenAI models and their maximum number of tokens.
"""

OpenAIChatModels = {k: v for k, v in OpenAIModels.items() if k.startswith("gpt-")}
"""
    List of OpenAI chat models and their maximum number of tokens.
"""
