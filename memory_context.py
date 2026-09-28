from dataclasses import dataclass

import requests
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime 
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

@dataclass
class Context:
    user_id: str

@dataclass
class ResponseFormat:
    summary : str
    temperature_celcius: float
    temperature_fareniheit: float
    humidity: float

@tool('get_weather', description='Return weather information for a given city', return_direct=False)
def get_weather(city: str):
    response = requests.get(f'https://wttr.in/{city}?format=j1')
    return response.json()

@tool('locate_user', description='return the location of user based on the context')
def locate_user(runtime: ToolRuntime[Context]):
    match runtime.context.user_id:
        case 'ABC123':
            return 'Exeter'
        case 'XYZ456':
            return 'London'
        case _:
            return 'Unkown'
        
model = init_chat_model(model='claude-sonnet-5')

checkpointer = InMemorySaver()

agent = create_agent(
    model = model,
    tools = [get_weather, locate_user],
    system_prompt = 'you are a helpful assistant, who is always cracking jokes but remains helpful.',
    context_schema= Context,
    response_format = ResponseFormat,
    checkpointer = checkpointer

)

config = {'configurable': {'thread_id': 1}}

response = agent.invoke({
    'messages':[{'role': 'user', 'content':'what is the weather in?'}]},
    config = config,
    context = Context(user_id='ABC123')

)

# print(response['structured_response'])
print(response['structured_response'].summary)
# print(response['structured_response'].temperature_celcius)

# response should have been different but it's somw how getting the location 
config = {'configurable': {'thread_id': 2}}

response = agent.invoke({
    'messages':[{'role':'user', 'content': 'is this usual?'}]
},
    config = config,
    context = Context(user_id='ABC123')
)

print(response['structured_response'].summary)
