import requests
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

model = init_chat_model(
    model = 'mistral', # works only for mistal when you want to see the reply while it is being generated
    temperature = 0.1
)

for chunk in model.stream('Hey! What is python?'):
    print(chunk.text, end='', flush=True)