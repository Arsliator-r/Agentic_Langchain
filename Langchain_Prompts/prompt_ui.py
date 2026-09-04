import streamlit as st
from langchain_openrouter import ChatOpenRouter
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

chat_model = ChatOpenRouter(model="anthropic/claude-sonnet-4.5", temperature=0.7, max_tokens=500)

st.title("AI Chat Interface")

paper_input = st.selectbox("Select Research Paper Name", ["Attention is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"])

style_input = st.selectbox("Select Style", [ "Beginner-Friendly", "Code-Oriented", "Technical", "Mathematical"])

length_input = st.selectbox("Select Length", ["Short (1-2 paragraphs)", "Medium (3-4 paragraphs)", "Long (5+ paragraphs)"])

prompt_template = PromptTemplate(
    input_variables=["paper_name", "style", "length"],
    template="""
You are an AI assistant that is expert in summarizing research papers titled '{paper_name}' in a following specifiactions:
Explanation Style: {style}
Length: {length}
1. Mathematical Details: 
    - Include relevant mathematical equations, derivations, and proofs where applicable.
    Explain the mathematical concepts using simple, intuitive code snippets wherever possible.
2. Analogies
    - Use reliable analogies to simplify complex concepts and make them more relatable.
If certain information is not available in the paper, please indicate that it is not available instead of guessing or hallucinating.
"""
)

prompt = prompt_template.invoke(
    {
        "paper_name": paper_input,
        "style": style_input,
        "length": length_input
    }
)


if st.button("Submit"):
    result = chat_model.invoke(prompt)
    st.subheader("Summary:")
    st.write(result.content)