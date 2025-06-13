from fastapi import FastAPI, Request
from pydantic import BaseModel
from dotenv import load_dotenv
import openai
import os
from utils import answer_query

load_dotenv()  # Loads variables from .env into environment

openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/ask")
async def ask(query: Query):
    answer = answer_query(query.question)
    return {"answer": answer}