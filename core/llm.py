"""LLM utility module"""

from os import environ as env

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

# from langchain_huggingface import HuggingFaceEmbeddings

from core.utils import SINGLETON_MANAGER
from core.configs import CONFIGS_LLM


def get_chat_llm() -> AzureChatOpenAI:
    """Get AzureChatOpenAI instance"""
    key = "AzureChatOpenAI"
    if not hasattr(SINGLETON_MANAGER, key):
        llm = AzureChatOpenAI(
            # azure_endpoint=env["AZURE_OPENAI_ENDPOINT"],
            # deployment_name=env["AZURE_OPENAI_LLM_DEPLOYMENT_NAME"],
            # openai_api_version=env["AZURE_OPENAI_API_VERSION"],
            # openai_api_key=env["AZURE_OPENAI_API_KEY"],
            # model=env["AZURE_OPENAI_LLM_MODEL"],
            temperature=CONFIGS_LLM.llm.temperature,
        )
        setattr(SINGLETON_MANAGER, key, llm)
    return getattr(SINGLETON_MANAGER, key)


def get_embeddings(provider: str = "azure"):

    match provider:
        case "azure":
            return get_azure_embeddings()

        # case "huggingface":
        #     return get_huggingface_embeddings()

        case _:
            raise ValueError(f"Invalid embeddings_model: {provider}")


def get_azure_embeddings() -> AzureOpenAIEmbeddings:
    """Get AzureEmbeddings instance"""
    key = "AzureOpenAIEmbeddings"
    if not hasattr(SINGLETON_MANAGER, key):
        embeddings = AzureOpenAIEmbeddings(
            # azure_endpoint=env["AZURE_OPENAI_ENDPOINT"],
            # deployment_name=env["AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME"],
            # openai_api_version=env["AZURE_OPENAI_API_VERSION"],
            # openai_api_key=env["AZURE_OPENAI_API_KEY"],
            # model=env["AZURE_OPENAI_EMBEDDINGS_MODEL"],
        )
        setattr(SINGLETON_MANAGER, key, embeddings)
    return getattr(SINGLETON_MANAGER, key)


# def get_huggingface_embeddings():
#     """Get HuggingFace instance"""
#     key = "HuggingFaceEmbeddings"
#     if not hasattr(SINGLETON_MANAGER, key):
#         embeddings = HuggingFaceEmbeddings(
#             model_name=CFGS["chain"].huggingface.embeddings_model,
#             model_kwargs={"device": "cuda"},
#         )
#         setattr(SINGLETON_MANAGER, key, embeddings)
#     return getattr(SINGLETON_MANAGER, key)


CHAT_LLM = get_chat_llm()
EMBEDDINGS = get_embeddings(provider="azure")

print(EMBEDDINGS.model_name)
