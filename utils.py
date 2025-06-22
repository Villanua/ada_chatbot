import os
import logging
from dotenv import load_dotenv

load_dotenv()

# Configure logger to use the basic logging format
logger = logging.getLogger(__name__)
if not logger.hasHandlers():  # Only add handler if no handlers exist
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s:%(filename)s:%(lineno)d:%(message)s')  # Format with source info
    handler.setFormatter(formatter)
    logger.addHandler(handler)
if eval(os.environ.get("INGESTION_DEBUG", "False")):
    logger.setLevel(logging.DEBUG)


def load_chat_model():
    """
    Retrieve a chat model instance based on the cloud platform environment.

    This function dynamically selects and initializes a chat model depending on the 
    cloud platform specified in the "CLOUD_PLATFORM" environment variable. 
    If the platform is set to "AZURE", the function configures and returns an 
    AzureChatOpenAI model using the `langchain_openai` library. Otherwise, it defaults 
    to returning a ChatVertexAI model configured with safety settings from 
    `langchain_google_vertexai`.

    Returns:
        An instance of either AzureChatOpenAI or ChatVertexAI, configured with parameters 
        retrieved from environment variables.

    Environment Variables:
        - CLOUD_PLATFORM: Determines which cloud platform to use ("AZURE" or other).
        - AZURE_OPENAI_API_KEY: API key for Azure OpenAI.
        - AZURE_OPENAI_ENDPOINT: Endpoint URL for Azure OpenAI.
        - AZURE_OPENAI_API_VERSION: API version for Azure OpenAI.
        - AZURE_OPENAI_CHAT_VERSION: Chat deployment version for Azure OpenAI.
        - MODEL_NAME_VERTEX: Model name for Google Vertex AI.
        - GOOGLE_PROJECT: Google Cloud project ID.
    """

    if os.getenv("CLOUD_PLATAFORM") == "AZURE":
        logger.debug("Getting AzureChatOpenAI")
        from langchain_openai import AzureChatOpenAI

        model = AzureChatOpenAI(
            max_tokens=10000,
            temperature=0,
            request_timeout=500,
            max_retries=10,
            deployment_name=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
            openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        )
        
    else:
        logger.debug("Getting ChatVertexAI")
        from langchain_google_vertexai import ChatVertexAI
        from vertexai.generative_models import HarmCategory, HarmBlockThreshold

        safety_settings = {
            HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        }
        model = ChatVertexAI(
            max_output_tokens = 5000,
            temperature = 0,
            model_name=os.getenv("MODEL_NAME_VERTEX"),
            safety_settings=safety_settings,
            project=os.getenv("GOOGLE_PROJECT")
        )
        
    return model



def load_embeddings_model():
    """
    Retrieve an embeddings model instance based on the cloud platform environment.

    This function dynamically selects and initializes an embeddings generator depending 
    on the cloud platform specified in the "CLOUD_PLATFORM" environment variable. 
    If the platform is set to "AZURE", it configures and returns an AzureOpenAIEmbeddings 
    instance from the `langchain_openai` library. Otherwise, it defaults to returning a 
    VertexAIEmbeddings instance from the `langchain_google_vertexai` library.

    Returns:
        An instance of either AzureOpenAIEmbeddings or VertexAIEmbeddings, configured with 
        parameters retrieved from environment variables.

    Environment Variables:
        - CLOUD_PLATFORM: Determines which cloud platform to use ("AZURE" or other).
        - AZURE_OPENAI_API_KEY: API key for Azure OpenAI.
        - AZURE_OPENAI_ENDPOINT: Endpoint URL for Azure OpenAI.
        - AZURE_OPENAI_API_VERSION: API version for Azure OpenAI.
        - AZURE_OPENAI_EMBEDDING_VERSION: Embedding model version for Azure OpenAI.
        - MODEL_NAME_EMBEDDINGS: Model name for Google Vertex AI embeddings.
    """

    if os.getenv("CLOUD_PLATAFORM") == "AZURE":
        logger.debug("Getting AzureOpenAIEmbeddings")
        from langchain_openai import AzureOpenAIEmbeddings

        embeddings_generator = AzureOpenAIEmbeddings(
            model=os.getenv("AZURE_OPENAI_EMBEDDING_VERSION"),
            chunk_size=50,
            max_retries=10,
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION")
        )

    else:
        logger.debug("Getting VertexAIEmbeddings")
        from langchain_google_vertexai import VertexAIEmbeddings
        
        embeddings_generator = VertexAIEmbeddings(
            model_name=os.getenv("MODEL_NAME_EMBEDDINGS")
        )

    return embeddings_generator