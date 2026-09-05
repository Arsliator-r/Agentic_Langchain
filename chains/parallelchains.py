from langchain_openai import ChatOpenAI
from langchain_openrouter import ChatOpenRouter
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv

load_dotenv()

llm1 = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b", 
    task="text-generation",
)

llm2 = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
)

# model1 = ChatOpenRouter(model_name='google/gemma-4-31b-it:free', temperature=0.7, max_tokens=1000)
model1 = ChatHuggingFace(llm=llm1)
# model2 = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
model2 = ChatHuggingFace(llm=llm2)

prompt1 = PromptTemplate(
    template="Generate a short and simple notes on the following topic : {topic}",
    input_variables=["topic"],
)

prompt2 = PromptTemplate(
    template="Generate 5 short QnA from the following topic : {topic}",
    input_variables=["topic"],
)

prompt3 = PromptTemplate(
    template="Merge the provided notes and quiz into a single document /n Notes: {notes} \n Quiz: {quiz}",
    input_variables=["notes", "quiz"],
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    "notes": prompt1 | model1 | parser,
    "quiz": prompt2 | model2 | parser
})

merge_chain = prompt3 | model1 | parser

chain = parallel_chain | merge_chain

result = chain.invoke({"topic": "Python programming"})

# print("\nFinal Result:\n", result)

chain.get_graph().print_ascii()


