from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
)

parser = JsonOutputParser()

model = ChatHuggingFace(llm=llm)

prompt_template = PromptTemplate(
    input_variables=[],
    template="""Give me a name, age, and city of a fictional person \n {format_instruction}""",
    partial_variables={"format_instruction": parser.get_format_instructions()}
)

prompt = prompt_template.invoke({})

result = model.invoke(prompt)

final_result = parser.parse(result.content)

print("\nFinal Result:\n", final_result)