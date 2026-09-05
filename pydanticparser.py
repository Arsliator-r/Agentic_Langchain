from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b", 
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

class Person(BaseModel):
    name: str = Field(..., description="The name of the person")
    age: int = Field(gt=18, description="The age of the person")
    city: str = Field(..., description="The city where the person lives")

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    input_variables=["place"],
    template="""Give me a name, age, and city of a fictional {place} person \n {format_instruction}""",
    partial_variables={"format_instruction": parser.get_format_instructions()}
)

# prompt = template.invoke({"place": "American"})

# result = model.invoke(prompt)

# final_result = parser.parse(result.content)

# print("\nFinal Result:\n", final_result)
# print(type(final_result.age))

chain = template | model | parser

result = chain.invoke({"place": "American"})

print("\nFinal Result:\n", result)
