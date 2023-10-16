from typing import List, Optional

from ...core.atoms.tool import Tool
from ..atoms import OpenAIChatModels, OpenAIModel


class OpenAIChatModel(OpenAIModel):
    tools: List[Tool]

    def __init__(
        self, name: str = "gpt-3.5-turbo", *args, tools: Optional[List[Tool]], **kwargs
    ):
        """
        Creates an OpenAI Chat model, optionally with a set of tools it can use.

        Args:
            name (str, optional): The name of the model version being used.
            Defaults to "gpt-3.5-turbo".
            tools (List[Tool], optional): The tools the model can use. Defaults to None.
            streaming (bool, optional): Whether the model should be initialized as
            streaming. Defaults to False.
            max_response (int, optional): The maximum number of tokens the model can
            return. Defaults to 256.
            api_key (str, optional): The API key for the model. Defaults to the
            OPENAI_API_KEY environment variable.
            top_p (float, optional): The nucleus sampling probability. Defaults to 1.0.
            temperature (float, optional): The temperature of the model.
            Defaults to 1.0.
            n (int, optional): The number of completions to generate. Defaults to 1.
            stop (list, optional): The stop sequence(s) for the model. Defaults to None.
            presence_penalty (float, optional): The presence penalty for the model.
            Defaults to 0.0.
            frequency_penalty (float, optional): The frequency penalty for the model.
            Defaults to 0.0.
            logit_bias (dict, optional): The logit bias for the model. Defaults to None.
            suffix (str, optional): The suffix for the model. Defaults to None.
            logprobs (int, optional): The number of logprobs for the model.
            Defaults to None.
            echo (bool, optional): Echo back the prompt in addition to the completion.
            Defaults to False.
            best_of (int, optional): Generates `best_of` completions server-side and
            returns the "best" (the one with the lowest log probability per token).
            Defaults to 1.
        """
        if name not in OpenAIChatModels:
            raise ValueError(f"Invalid OpenAI chat model name: {name}")
        super().__init__(*args, **kwargs)
        self.tools = tools

    # TODO: Override _format_request to include tools as OpenAI functions
