import random

from dotenv import load_dotenv

from logging_config import logger
from SilverLingua.core.atoms import Notion, Tool, prompt
from SilverLingua.openai import OpenAIChatAgent  # noqa
from SilverLingua.openai.atoms.role.chat import OpenAIChatRole
from SilverLingua.util import timeit


# test
@timeit
@prompt
def fruit_prompt(fruits: list):
    """
    You are a helpful assistant that takes a list of fruit and gives information about their nutrition.

    LIST OF FRUIT:
    {% for fruit in fruits %}{{ fruit }}
    {% endfor %}
    """


logger.debug(fruit_prompt(["apple", "orange"]))


##########
def roll_dice(
    sides: int = 20,
    dice: int = 1,
    modifier: int = 0,
    advantage: bool = False,
    disadvantage: bool = False,
):
    """
    Rolls a number of dice with a given number of sides, optionally with a modifier and/or advantage/disadvantage. Returns `{result: int, rolls: int[]}`

    Args:
        sides: The number of sides on each die (default 20)
        dice: The number of dice to roll (default 1)
        modifier: The modifier to add to the roll total (default 0)
        advantage: Whether to roll with advantage (default False)
        disadvantage: Whether to roll with disadvantage (default False)
    """
    if advantage and disadvantage:
        raise ValueError("Can't roll with both advantage and disadvantage.")

    # Ensure at least two dice are rolled if advantage or disadvantage is specified
    if advantage or disadvantage:
        dice = max(2, dice)

    rolls = [random.randint(1, sides) for _ in range(dice)]

    if advantage:
        roll_result = max(rolls)
    elif disadvantage:
        roll_result = min(rolls)
    else:
        roll_result = sum(rolls)

    ret = {"result": roll_result + modifier, "rolls": rolls}
    print(f"Rolled {ret}")
    return ret


rd_tool = Tool(roll_dice)
logger.debug(rd_tool.name)

##########
load_dotenv()
agent = OpenAIChatAgent()  # ("gpt-4-1106-preview")
agent.add_tool(rd_tool)

print("About to start chat stream...")
stream = agent.stream(
    [
        Notion(
            "roll 1d20",
            str(OpenAIChatRole.HUMAN.value),
        )
    ]
)
for notion in stream:
    agent.idearium.append(notion)
    print(agent.idearium[-1])

r = agent.generate([Notion(":)", str(OpenAIChatRole.HUMAN.value))])
agent.idearium.append(r[0])
print(r[0])

# Start command line chat with agent
"""try:
    while True:
        message = input("You: ")

        if message == "exit":
            break

        response = agent.generate([Notion(message, str(OpenAIChatRole.HUMAN.value))])
        agent.idearium.append(response[0])
        print(response[0])
finally:
    print("Chat session ended.")"""
