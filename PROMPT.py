from huggingface_hub import hf_hub_download
from langchain_core.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.llms.llamacpp import LlamaCpp
from RAG import search
path_model = hf_hub_download(repo_id="bartowski/Gemma-2-Ataraxy-9B-GGUF", filename="Gemma-2-Ataraxy-9B-IQ2_M.gguf")
llm = LlamaCpp(
    model_path=path_model,
    max_tokens=512,
    n_ctx=4096,
    verbose=True,
)

PLAN_PROMPT_SEARCH = """Answer the following question correctly based on the provided context. The question could be tricky as well, so think step by step and answer it correctly.

Context: {search_result}

Question: {query}

Answer:"""
search_response_prompt = PromptTemplate.from_template(PLAN_PROMPT_SEARCH)
search_response = search_response_prompt | llm | StrOutputParser()
def run_search(query):
    search_result= search(query)
    print (search_result[0]['link'])
    verified_answer = search_response.invoke({"query": query,"search_result":search_result[0]['abstract']})
    return verified_answer