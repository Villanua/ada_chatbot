from pydantic import BaseModel, Field
from typing import Optional

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import PydanticOutputParser


class Answer(BaseModel):
    answer: str = Field(
        description="The detailed response generated based on the user's question and the provided context.",
        default="LLM error"
    )
    answered: bool = Field(
        description="Indicates whether the system has found information in the document that answers the user's question.",
        default=False
    )


def get_answer_question_chain(prompt_template, llm, chat_conversation_buffer, language):
    """
    This chain answers a question's user.
    """

    parser = PydanticOutputParser(pydantic_object=Answer)

    prompt = PromptTemplate(
        input_variables=["question", "context"], 
        template=prompt_template,
        partial_variables={
            "chat_history": chat_conversation_buffer,
            "format_instructions": parser.get_format_instructions(),
            'language': language}
        )
    
    question_answer_chain: RunnableSequence = prompt | llm | parser

    return question_answer_chain