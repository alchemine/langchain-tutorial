from langchain import hub
from langchain.agents import create_react_agent
from langchain.agents import AgentExecutor
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

from src.utils import get_chat_llm_client
from tools_2 import get_tools

# from langchain.agents.react.output_parser import ReActOutputParser


import re
from typing import Union

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.exceptions import OutputParserException

from langchain.agents.agent import AgentOutputParser


class ReActOutputParser(AgentOutputParser):
    """Output parser for the ReAct agent."""

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        # action_prefix = "Action: "
        # if not text.strip().split("\n")[-1].startswith(action_prefix):
        #     raise OutputParserException(f"Could not parse LLM Output: {text}")
        # action_block = text.strip().split("\n")[-1]
        # action_block = text.split("\n")[2]

        # action_block = text.split("\n")[2]
        # action_str = action_block[len(action_prefix) :]
        # # Parse out the action and the directive.
        # re_matches = re.search(r"(.*?)\[(.*?)\]", action_str)
        # if re_matches is None:
        #     raise OutputParserException(
        #         f"Could not parse action directive: {action_str}"
        #     )
        # action, action_input = re_matches.group(1), re_matches.group(2)

        print(text)
        pattern = r"Action: (.*)\nAction Input:(.[\s\S]*)"
        match = re.search(pattern, text)
        try:
            if match is None:
                raise OutputParserException(f"Could not parse LLM Output: {text}")
        except Exception as e:
            print(e)
            print("text:", text)
            exit()

        action, action_input = match.groups()
        if action == "Finish":
            return AgentFinish({"output": action_input}, text)
        else:
            try:
                action_input = eval(action_input)
            except Exception as e:
                print(e)
                print(100 * "=")
                print("text:", text)
                print(100 * "=")
                exit()
            return AgentAction(action, action_input, text)


SYSTEM_PROMPT = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("user", "Question: {input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)

# prompt1 = hub.pull("hwchase17/react")
# prompt1.pretty_print()

# """
# Answer the following questions as best you can. You have access to the following tools:

# {tools}

# Use the following format:

# Question: the input question you must answer
# Thought: you should always think about what to do
# Action: the action to take, should be one of [{tool_names}]
# Action Input: the input to the action
# Observation: the result of the action
# ... (this Thought/Action/Action Input/Observation can repeat N times)
# Thought: I now know the final answer
# Final Answer: the final answer to the original input question

# Begin!

# Question: {input}
# Thought:{agent_scratchpad}
# """

# # instructions = """You are an agent designed to write and execute python code to answer questions.
# # You have access to a python REPL, which you can use to execute python code.
# # If you get an error, debug your code and try again.
# # Only use the output of your code to answer the question.
# # You might know the answer without running any code, but you should still run the code to get the answer.
# # If it does not seem like you can write code to answer the question, just return "I don't know" as the answer.
# # """
instructions = """Answer the following questions as best you can.
You have access to the following tools:
"""
base_prompt = hub.pull("langchain-ai/react-agent-template")
prompt2 = base_prompt.partial(instructions=instructions)
prompt2.pretty_print()

agent = create_react_agent(
    llm=get_chat_llm_client(),
    tools=get_tools(),
    prompt=prompt2,
    output_parser=ReActOutputParser(),
)
rst = agent.invoke({"input": "ESG 강의 찾아줘", "intermediate_steps": []})
print(rst)

# agent_executor = AgentExecutor(
#     agent=agent,
#     tools=get_tools(),
#     verbose=True,
#     max_iterations=3,
# )
# agent_executor.invoke({"input": "ESG 강의 찾아줘"})