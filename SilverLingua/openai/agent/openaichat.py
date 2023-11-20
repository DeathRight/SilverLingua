from typing import List, Union

from SilverLingua.core.agent import Agent
from SilverLingua.core.atoms import ChatRole, Idearium, Notion, Tool
from SilverLingua.core.atoms.tool import FunctionCall, FunctionResponse

from ..atoms import OpenAIChatModels, OpenAIModel


class OpenAIChatAgent(Agent):
    """
    An agent that uses the OpenAI chat completion API.
    """

    _tools: List[Tool]
    _idearium: Idearium
    _model: OpenAIModel

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        api_key: str = "",
        idearium: Union[Idearium, None] = None,
        tools: Union[List[Tool], None] = None,
        *args,
        **kwargs,
    ) -> None:
        """
        Initializes an OpenAIChatAgent.

        In addition to the explicit args, also accepts all args from OpenAIModel.

        Args:
            model (str): The name of the model to use.
            api_key (st, optional): The API key to use. Defaults to "".
            idearium (Union[Idearium, None], optional): The idearium to use.
            Defaults to None.
            tools (Union[List[Tool], None], optional): The tools to use.
            Defaults to None.
            streaming (bool, optional): Whether to initialize the model as streaming.
            Defaults to False.
            max_response (int, optional): The maximum number of tokens to return.
            Defaults to 256.

        Raises:
            ValueError: If the model name is invalid.
        """
        if model not in OpenAIChatModels:
            raise ValueError(f"Invalid model name: {model}")
        self._model = model
        self._tools = tools if tools is not None else []

        self._model = OpenAIModel(model, api_key=api_key, *args, **kwargs)
        self._idearium = (
            idearium
            if idearium is not None
            else Idearium(self._model.tokenizer, self._model.max_tokens)
        )

    @property
    def model(self) -> OpenAIModel:
        """
        The model used by the agent.
        """
        return self._model

    @property
    def idearium(self) -> Idearium:
        """
        The idearium used by the agent.
        """
        return self._idearium

    @property
    def tools(self) -> List[Tool]:
        """
        The tools used by the agent.
        """
        return self._tools

    def generate(self, messages: Union[Idearium, List[Notion]]) -> List[Notion]:
        response = self._model.generate(messages)[0]

        # Check responses for function calls
        if response.chat_role == ChatRole.TOOL_CALL:
            function_call = FunctionCall.from_json(response.content)

            # Find the tool
            tool = None
            if self.tools is not None:
                for t in self.tools:
                    if t.name == function_call.name:
                        tool = t
                        break

            # Call the tool
            if tool is not None:
                tool_response = tool()
            else:
                return [
                    Notion(
                        ChatRole.TOOL_RESPONSE,
                        FunctionResponse(None, "Tool not found").to_json(),
                    )
                ]
