from datetime import UTC, datetime
import os
from dotenv import load_dotenv
from pymongo import MongoClient
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_groq import ChatGroq

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
mongodb_url = os.getenv("MONGODB_URL")

client = MongoClient(mongodb_url)
db = client["chatbot_sa"]
collection = db["Student_Users"]
app = FastAPI()


class ChatRequest(BaseModel):
    user_id: str
    question: str


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful academic study assistant. "
            "Use prior conversation context when relevant. "
            "Answer study-related questions clearly and concisely. "
            "If unsure, say you don't know.",
        ),  
        ("placeholder", "{history}"),
        ("user", "{question}"),
    ]
)


llm = ChatGroq(api_key=groq_api_key, model="openai/gpt-oss-20b")
chain = prompt | llm


def get_history(user_id):
    chats = collection.find({"user_id": user_id}).sort("timestamp", 1)
    history = []
    for chat in chats:
        history.append((chat["role"], chat["message"]))
    return history


@app.get("/")
def home():
    return {"message": "Welcome to the Student assistant Chatbot API"}


@app.post("/chat")
def chat(request: ChatRequest):

    history = get_history(user_id=request.user_id)

    response = chain.invoke({"history": history, "question": request.question})

    collection.insert_one(
        {
            "user_id": request.user_id,
            "role": "user",
            "message": request.question,
            "timestamp": datetime.now(UTC),
        }
    )

    collection.insert_one(
        {
            "user_id": request.user_id,
            "role": "assistant",
            "message": response.content,
            "timestamp": datetime.now(UTC),
        }
    )

    return {"response": response.content}
