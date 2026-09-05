from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b", 
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

class Sentiment(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"] = Field(..., description="The sentiment of the product review")

sentiment_parser = PydanticOutputParser(pydantic_object=Sentiment)

prompt1 = PromptTemplate(
    template="Extract the sentiment of the following product review: {review}, \n {format_instruction}",
    input_variables=["review"],
    partial_variables={"format_instruction": sentiment_parser.get_format_instructions()}
) 

classifier_chain = prompt1 | model | sentiment_parser

prompt2 = PromptTemplate(
    template="Write an appropriate response to the following positive product review: {review}",
    input_variables=["review"]
)

chain2 = prompt2 | model | parser

prompt3 = PromptTemplate(
    template="Write an appropriate response to the following negative product review: {review}",
    input_variables=["review"]
)

chain3 = prompt3 | model | parser

prompt4 = PromptTemplate(
    template="Write an appropriate response to the following neutral product review: {review}",
    input_variables=["review"]
)

chain4 = prompt4 | model | parser

branch_chain = RunnableBranch(
    (lambda x:x.sentiment == 'positive', chain2),
    (lambda x:x.sentiment == 'negative', chain3),
    (lambda x:x.sentiment == 'neutral', chain4),
    RunnableLambda(lambda _: "could not classify the sentiment")
)

chain = classifier_chain | branch_chain

print(chain.invoke({"review": "Awfull experience, i mean the product is pkay, but the delivery, support is just terrible shit!!!"}))


