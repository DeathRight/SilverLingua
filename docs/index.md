# SilverLingua

A type-safe framework for building AI agents using atomic design patterns and hierarchical memory.

## Why SilverLingua?

Unlike LangChain's complex chains and callbacks, SilverLingua provides:

- **Type Safety**: Every component is a Pydantic model - no more runtime surprises
- **Atomic Design**: Build complex agents from simple, composable pieces
- **Clean Interfaces**: Consistent patterns across the entire framework
- **Provider Agnostic**: Switch between LLMs without changing your agent logic

## Core Architecture

### Memory System

```python
# Base memory unit - simple and type-safe
class Memory(BaseModel):
    content: str

# Role-aware memory with validation
class Notion(Memory):
    role: str
    persistent: bool = False
```

### Idearium - Extensible Memory Management

```python
# LangChain's ConversationBufferMemory - Complex inheritance, mixed concerns
class ConversationBufferMemory(BaseChatMemory):
    return_messages: bool = False
    human_prefix: str = "Human"
    ai_prefix: str = "AI"
    memory_key: str = "history"

    @property
    def buffer(self) -> Any:  # Type hints lost
        if self.return_messages: ...
        else: ...

# SilverLingua's Idearium - Clean, composable, token-aware
class Idearium(BaseModel):
    tokenizer: Tokenizer
    max_tokens: int
    notions: List[Notion]
    persistent_indices: set

    def _trim(self):
        """Extension point for custom trimming strategies"""
        while self.total_tokens > self.max_tokens:
            # Default implementation
            pass

# Easy to extend for different memory strategies
class SummarizingIdearium(Idearium):
    def _trim(self):
        """Custom trimming with automatic summarization"""
        if self.total_tokens > self.max_tokens:
            # Summarize oldest non-persistent memories
            summary = self.summarize_notions(self.get_trim_candidates())
            self.replace_with_summary(summary)

class RAGIdearium(Idearium):
    def __init__(self, vector_store, **kwargs):
        super().__init__(**kwargs)
        self.vector_store = vector_store

    def _trim(self):
        """Store trimmed memories in vector store"""
        for notion in self.get_trim_candidates():
            self.vector_store.add(notion.content)
        super()._trim()

    def get_context(self, query: str) -> List[Notion]:
        """Retrieve relevant memories"""
        return self.vector_store.similarity_search(query)
```

### Provider Integration

```python
# Clean model interface
class OpenAIModel:
    def generate(self, prompt: str) -> str:
        # OpenAI-specific implementation
        pass

# Consistent agent pattern
class OpenAIChatAgent(Agent):
    def _bind_tools(self) -> None:
        # Automatic OpenAI function calling format
        self.model.completion_params.tools = [
            {"type": "function", "function": tool.description}
            for tool in self.tools
        ]
```

## Why This Matters

### LangChain Memory

```python
# Complex setup, multiple inheritance, mixed concerns
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="output",
    input_key="input",
)
chain = LLMChain(
    llm=llm,
    memory=memory,
    prompt=prompt,
    verbose=True
)
```

### SilverLingua Memory

```python
# Clean, composable, type-safe
idearium = RAGIdearium(
    vector_store=vector_store,
    tokenizer=tokenizer,
    max_tokens=4096
)

# Easy to use
idearium.append(Notion("Important fact", persistent=True))
idearium.extend(new_memories)  # Auto-manages tokens

# Automatic context management
relevant_context = idearium.get_context("query")
```

## Key Benefits

1. **Extensible Memory**

   - Clear extension points (`_trim`, etc.)
   - Type-safe customization
   - Easy integration with external systems

2. **Token Management**

   - Automatic token tracking
   - Smart memory trimming
   - Persistent memory support

3. **Clean Architecture**
   - Single responsibility principle
   - Consistent interfaces
   - Type-safe operations

## Documentation

- [Installation](getting-started/installation.md)
- [API Reference](api/core/index.md)
- [Examples](getting-started/quickstart.md)
