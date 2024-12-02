# Decorators

SilverLingua provides several decorators to enhance function behavior and make common patterns easier to implement.

## Prompt Decorator

The prompt decorator transforms a function's docstring into a Jinja2 template, making it easy to create dynamic prompts.

::: SilverLingua.core.atoms.prompt.prompt

## Tool Decorator

The tool decorator converts a function into a Tool instance, allowing it to be used as a callable tool by AI agents.

::: SilverLingua.core.atoms.tool.decorator.tool
