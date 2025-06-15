import os
import yaml
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from constants import *

with open("prompt.yaml", 'r') as f:
    prompt_data = yaml.safe_load(f)

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_data['template']
)

llm = ChatOpenAI(model_name=MODEL_NAME, temperature=TEMPERATURE)
embedding = OpenAIEmbeddings(model=EMBEDDING_MODEL)
vectorstore = FAISS.load_local(VECTOR_STORE_PATH, embedding,allow_dangerous_deserialization=True)

retriever = vectorstore.as_retriever()
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
rag_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory,
                                                combine_docs_chain_kwargs = {"prompt":prompt})
                                                  
                                                  
def answer_query(question: str):
    return rag_chain.run(question)
