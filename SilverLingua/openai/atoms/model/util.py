# List of OpenAI models and their maximum number of tokens.
OpenAIModels = {
    "text-embedding-ada-002": 8191,
    "text-davinci-003": 4097,
    "gpt-3.5-turbo": 4097,
    "gpt-3.5-turbo-16k": 16385,
    "gpt-4": 8192,
    "gpt-4-32k": 32768,
    "gpt-4-1106-preview": 128000,
}

OpenAIChatModels = {k: v for k, v in OpenAIModels.items() if k.startswith("gpt-")}
