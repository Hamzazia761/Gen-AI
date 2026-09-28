import requests
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, AIMessage, SystemMessage # not needed for the comment below.

load_dotenv()

model = init_chat_model(
    model = 'claude-sonnet-5'
)
# 2
# for history
conversation = [
    SystemMessage('you are a helpful assistant that answers question regarding programming.'),
    HumanMessage('What is Python?'),
    AIMessage('Python is a high-level, interpreted programming language known for its clean, readable syntax and versatility.'),
    HumanMessage('When was it released?')
    ]
# 2

# response = model.invoke('Hello, What is Python?')
response = model.invoke(conversation)


print(response.content)

