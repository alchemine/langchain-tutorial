# from langchain.chains.openai_functions.tagging import _TAGGING_TEMPLATE
TAGGING_TEMPLATE = """Extract the desired information from the following passage.

Only extract the properties mentioned in the 'information_extraction' function.

Passage:
{input}
"""


# https://python.langchain.com/docs/modules/memory/custom_memory
ENTITIES_INPUT_TEMPLATE = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know. You are provided with information about entities the Human mentions, if relevant.

Relevant entity information:
{entities}

Conversation:
Human: {input}
AI:"""


# https://www.deeplearning.ai/short-courses/functions-tools-agents-langchain/
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
