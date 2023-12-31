import logging
from enum import Enum
from typing import List, Optional, Type

from pydantic import BaseModel, field_validator

from .core.atoms.role import ChatRole, ReactRole
from .core.atoms.tool import Tool

logger = logging.getLogger(__name__)


class Module(BaseModel):
    """
    A module that can be loaded into SilverLingua.

    Attributes:
        name: The name of the module.
        description: A description of the module.
        version: The version of the module.
        tools: The tools in the module.
        chat_roles: The chat roles in the module.
        react_roles: The react roles in the module.
    """

    name: str
    description: str
    version: str
    tools: List[Tool]
    chat_roles: List[type[ChatRole]]
    react_roles: List[type[ReactRole]]

    @field_validator("chat_roles", mode="plain")
    def check_chat_roles(cls, v):
        for role in v:
            if not issubclass(role, Enum):
                raise TypeError("Expected an enum")
            else:
                for member in role:
                    if (
                        not ChatRole[member.name]
                        or member.value != ChatRole[member.name].value
                    ):
                        raise TypeError("members must match ChatRole members.")
        return v

    @field_validator("react_roles", mode="plain")
    def check_react_roles(cls, v):
        for role in v:
            if not issubclass(role, Enum):
                raise TypeError("Expected an enum")
            else:
                for member in role:
                    if (
                        not ReactRole[member.name]
                        or member.value != ReactRole[member.name].value
                    ):
                        raise TypeError("members must match ReactRole members.")
        return v

    def __init__(
        self,
        name: str,
        description: str,
        version: str,
        tools: List[Tool],
        chat_roles: List[type[ChatRole]],
        react_roles: List[type[ReactRole]],
        **kwargs,
    ):
        super().__init__(
            name=name,
            description=description,
            version=version,
            tools=tools,
            chat_roles=chat_roles,
            react_roles=react_roles,
            **kwargs,
        )

        Config.register_module(self)


class Config:
    modules: List[Module] = []
    chat_roles: List[type[ChatRole]] = [ChatRole]
    react_roles: List[type[ReactRole]] = [ReactRole]
    tools: List[Tool] = []

    @classmethod
    def get_chat_role(self, role: str) -> Optional[ChatRole]:
        """
        Attempts to get the standardized ChatRole enum from 'role'.
        If not, returns None.

        This is usually used internally for maintaining
        consistency in Notions across different LLM backends.
        """
        for enum_class in self.chat_roles:
            for enum_member in enum_class:
                if str(enum_member.value).lower() == str(role).lower():
                    return ChatRole[enum_member.name]
        return None

    @classmethod
    def get_react_role(self, role: str) -> Optional[ReactRole]:
        """
        Attempts to get the standardized ReactRole enum from 'role'.
        If not, returns None.

        This is usually used internally for maintaining
        consistency in Notions across different LLM backends.
        """
        for enum_class in self.react_roles:
            for enum_member in enum_class:
                if enum_member.value == role:
                    return ReactRole[enum_member.name]
        return None

    @classmethod
    def get_tool(self, name: str) -> Optional[Tool]:
        """
        Attempts to get the tool with the given name.
        If not, returns None.
        """
        for tool in self.tools:
            if tool.name == name:
                return tool
        return None

    @classmethod
    def add_tool(self, tool: Tool) -> None:
        """
        Adds a tool to the config.
        """
        self.tools.append(tool)

    @classmethod
    def add_chat_role(self, role: Type[Enum]) -> None:
        """
        Adds a chat role to the config.
        """
        if not issubclass(role, Enum):
            raise TypeError("Expected an enum")
        self.chat_roles.append(role)

    @classmethod
    def add_react_role(self, role: Type[Enum]) -> None:
        """
        Adds a react role to the config.
        """
        if not issubclass(role, Enum):
            raise TypeError("Expected an enum")
        self.react_roles.append(role)

    @classmethod
    def register_module(self, module: Module) -> None:
        """
        Registers a module.
        """
        self.modules.append(module)
        for tool in module.tools:
            self.add_tool(tool)
        for chat_role in module.chat_roles:
            self.add_chat_role(chat_role)
        for react_role in module.react_roles:
            self.add_react_role(react_role)

        logger.debug(
            f'Registered module {module.name}@{module.version}: "{module.description}"'
        )
