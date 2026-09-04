from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()  # Load environment variables from .env file

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

template1 = PromptTemplate(
    input_variables=['topic'],
    template="""Write a deatiled paragraph on the following topic highlighting almost every aspect {topic}"""
)



template2 = PromptTemplate(
    input_variables=['paragraph'],
    template="""Write a 500 words summary of the following {paragraph}"""
)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({"topic": "Artificial Intelligence"})

print("\nSummary:\n", result)

# prompt1 = template1.invoke({"topic": "Artificial Intelligence"})
# result1 = model.invoke(prompt1)

# prompt2 = template2.invoke({"paragraph": result1.content})
# result2 = model.invoke(prompt2)

# print("Detailed Paragraph:\n", result1.content)
# print("\nSummary:\n", result2.content)
