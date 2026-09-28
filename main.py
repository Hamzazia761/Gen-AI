import requests
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.tools import tool

load_dotenv()

@tool('get_weather', description='Return weather information for a given city', return_direct=False)
def get_weather(city: str):
    response = requests.get(f'https://wttr.in/{city}?format=j1')
    return response.json()

agent = create_agent(
    model = 'claude-sonnet-5',
    tools = [get_weather],
    system_prompt = 'you are a helpful assistant, who is always cracking jokes but remains helpful.'
)

response = agent.invoke({
    'messages':[{'role': 'user', 'content':'what is the weather in exeter?'}]
}
)

# print(response['messages'][-1].content)
print(response)
