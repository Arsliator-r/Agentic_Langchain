from langchain_openrouter import ChatOpenRouter
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

chat_model = ChatOpenRouter(model = "openai/gpt-4o-mini", temperature = 1.8)
result = chat_model.invoke("Suggest me a good name for a new AI startup that focuses on natural language processing.")

print(result.content)