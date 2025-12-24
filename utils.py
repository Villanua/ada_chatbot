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
    cloud platform specified in the "CLOUD_PLATFORM" environment variable. It defaults
    to returning a ChatVertexAI model configured with safety settings from
    `langchain_google_vertexai`.

    The ChatVertexAI model is configured with specific safety settings to allow
    unrestricted content generation. It uses parameters retrieved from environment
    variables for customization.

    Returns:
        An instance of ChatVertexAI, configured with parameters retrieved from
        environment variables.

    Environment Variables:
        - MODEL_NAME_VERTEX: Model name for Google Vertex AI.
        - GOOGLE_PROJECT: Google Cloud project ID.
    """

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
        project=os.getenv("PROJECT_ID"),
        streaming=True,
    )
        
    return model



def load_embeddings_model():
    """
    Retrieve an embeddings model instance based on the cloud platform environment.

    This function dynamically selects and initializes an embeddings generator depending
    on the cloud platform specified in the "CLOUD_PLATFORM" environment variable. It
    defaults to returning a VertexAIEmbeddings instance from the `langchain_google_vertexai`
    library.

    The VertexAIEmbeddings model is configured using parameters retrieved from environment
    variables for customization.

    Returns:
        An instance of VertexAIEmbeddings, configured with parameters retrieved from
        environment variables.

    Environment Variables:
        - MODEL_NAME_EMBEDDINGS: Model name for Google Vertex AI embeddings.
    """

    logger.debug("Getting VertexAIEmbeddings")
    from langchain_google_vertexai import VertexAIEmbeddings
    
    embeddings_generator = VertexAIEmbeddings(
        model_name=os.getenv("MODEL_NAME_EMBEDDINGS")
    )

    return embeddings_generator