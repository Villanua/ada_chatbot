import logging
import json
from typing import Dict, Any
import logging

from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS

from ada_chatbot.state import State
from ada_chatbot.chains.answer_question_chains import get_answer_question_chain

logging.basicConfig(level=logging.DEBUG)


def answer_question(state: State) -> Dict[str, Any]:
    """
    Generate an answer to a question using relevant documents, conversation history, and a language model.

    This function answers a user's question by first preparing the conversation history and detecting the 
    language of the query. It then uses the conversation memory and document embeddings to provide context 
    to a language model. The language model is prompted to generate an answer in the detected language, 
    with relevant context from the selected documents. The answer and context are then saved in the database.

    Parameters:
        state (State): Contains the TODO

    Returns:
        Dict[str, Any]: TODO
    """

    logging.info(" ########## NODE ########## - ANSWER QUESTION\n")

    question = state["question"]
    documents = state["documents"]
    target_documents_id = state["target_documents_id"]
    prompts = state["prompts"]
    chat_conversation_json = state["chat"]
    llm = state["llm"]
    embeddings_generator = state["embeddings"]
    language = state["language"]
    k = state["config"].search_kwargs["k"]
    fetch_k = state['config'].search_kwargs["fetch_k"]

    if not language:
        language = "spanish"
        logging.info(f"Automatic language: {language}")

    # Define and generate the conversation buffer memory
    logging.debug("Creating conversation buffer memory")
    chat_conversation = json.loads(chat_conversation_json)
    chat_conversation_buffer = ConversationBufferMemory(return_messages=True, memory_key='chat_history', output_key='answer')

    logging.debug("Generating conversation buffer memory")
    str_conversation = ""
    if len(chat_conversation) > 0:
        for message in chat_conversation:
            chat_conversation_buffer.chat_memory.add_user_message(message['input'])
            chat_conversation_buffer.chat_memory.add_ai_message(message['output'])
            str_conversation += message['input'] + "\n\n" + message['output'] + "\n\n"

    str_conversation += question.question

    language_sentence = f"Your final answer has to be STRICTLY in {language}. ".format(language)
    prompt_template = prompts.objects.filter(name="answer_question").order_by('-created_date').first().prompt
    prompt_template = prompt_template + language_sentence

    # Fiter documents based on summary selection
    target_documents = [doc for doc in documents if doc['id'] in target_documents_id]
    valid_answer = None
    
    # Getting FAISS embeddings store
    logging.debug("Answering questions")

    answer_contexts = ""

    for n_doc in range(0, len(target_documents)):
        
        try: #TODO: Cambiar esto, comprobar si admite que haya el 'allow_dangerous_deserialization' cuando no es peligroso
            vector_store = FAISS.deserialize_from_bytes(
                target_documents[n_doc]['vectorstore'], 
                embeddings_generator,
                allow_dangerous_deserialization=True
            )
        except:
            vector_store = FAISS.deserialize_from_bytes(
                target_documents[n_doc]['vectorstore'], 
                embeddings_generator,
            )

        retrieved_docs = vector_store.similarity_search(
            query=question.enhanced_question,
            k=k,
            fetch_k=fetch_k
            )
    
        # Use chain to question the answer
        answer_question_chain = get_answer_question_chain(prompt_template, llm, chat_conversation_buffer, language)

        try:
            response = answer_question_chain.invoke({
                "question": question.enhanced_question,
                "context": retrieved_docs
                })

            answer = response.answer
            score = response.answered
        except:
            answer = ""
            score = False

        logging.info(f"Answer for document {target_documents[n_doc]['title']}: {score} {answer}")
        print(type(target_documents[n_doc]['title']))

        # Save answer in DB
        answer_db = Answer()
        answer_db.question = question
        answer_db.answer = answer
        answer_db.document = target_documents[n_doc]['path'] + "/" + target_documents[n_doc]['title']
        answer_db.valid = score
        answer_db.sources = target_documents[n_doc]['title']
        answer_db.save()

        if score:
            valid_answer = answer_db
            for doc in retrieved_docs:
                answer_contexts += doc.page_content + "\n\n"
    
    if not valid_answer:
        valid_answer = answer_db
    
    return {"answer": valid_answer, "answer_contexts": answer_contexts, "n_tries": state["n_tries"]+1}