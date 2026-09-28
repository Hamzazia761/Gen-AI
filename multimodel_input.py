# notebook for image data
from base64 import b64encode

from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage

load_dotenv()

model = init_chat_model(model='claude-sonnet-5')

# without human message
# message = {
#     'role': 'user',
#     'content': [
#         {'type': 'text', 'text': 'describe the contents of the image.'},
#         # {'type': 'image', 'url': 'https://kristenelizabethphotography.co/wp-content/uploads/2025/02/Senior-Photography-Minneapolis-Minnesota-132.jpg'}
#         {'type': 'image',
#          'base64': b64encode(open('group_picture.jpg', 'rb').read()).decode(),
#         'mime_type': 'image/jpeg'
#          }
#         # rb is reading byte mode
#     ]   
# }

# with Human Message
message = HumanMessage(
    content= [
        {'type': 'text', 'text': 'describe the contents of the image.'},
        # {'type': 'image', 'url': 'https://kristenelizabethphotography.co/wp-content/uploads/2025/02/Senior-Photography-Minneapolis-Minnesota-132.jpg'}
        {'type': 'image',
         'base64': b64encode(open('group_picture.jpg', 'rb').read()).decode(),
        'mime_type': 'image/jpeg'
         }
        # rb is reading byte mode
    ]   
)

response = model.invoke([message])

print(response.content)