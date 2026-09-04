from langchain_openrouter import ChatOpenRouter
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

chat_model = ChatOpenRouter(model = "anthropic/claude-sonnet-4.5", temperature = 0.7, max_tokens = 500)
result = chat_model.invoke("Suggest some financial freedom strategies for a 30-year-old individual with a stable job and moderate savings.")

print(result.content)