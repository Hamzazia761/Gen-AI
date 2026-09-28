from dataclasses import dataclass
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse, dynamic_prompt

load_dotenv()

@dataclass
class Context:
    user_role: str

@dynamic_prompt
def user_role_prompt(request: ModelRequest) -> str:
    user_role = request.runtime.context.user_role

    base_prompt = 'You are a helpful and very concise assistant.'

    match user_role:
        case 'expert':
            return f'{base_prompt} Provide detail technical response.'
        case 'beginner':
            return f'{base_prompt} Keep you explanation simple and basic.'
        case 'child':
            return f'{base_prompt} Explain everything as you are literally talking to a five year old child.'
        case _:
            return base_prompt
        

agent = create_agent(
    model='claude-sonnet-5',
    middleware = [user_role_prompt],
    context_schema = Context
)

response = agent.invoke({
    'message' : [{'role': 'user', 'content': 'Explain PCA.'}]
    },
    context = Context(user_role='child'))

print(response)