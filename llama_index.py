from dotenv import load_dotenv
from langchain.llms.base import LLM
import torch, textwrap, time, os
from llama_index import (
    PromptHealper, 
    SimpleDirectoryReader,
    LLMPredictor,
    ServiceContext,
    GPTListIndex
)
from transformers import pipeline
# from streamlit_chat import message

load_dotenv()


def timeit():
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            args = [str(arg) for arg in args]

            print(f"[{(end -start):.8f} seconds]: f({args}) -> {result}")
            return result
        return wrapper
    return decorator


max_tokens:int = 256

prompt_healper = PromptHealper(
    maxinput_size = 1024,
    num_output = max_tokens,
    max_chunk_overlap = 20
)

class LocalOPT(LLM):
    # model_name:str = "facebook/opt-iml-max-30b" # this is a 60gb model
    model_name:str = "facebook/opt-iml-1.3b" # this is a 2.63gb model
    pipeline = pipeline(
        "text-generation",
        model = model_name,
        # device:str = "cuda:0",
        model_kwargs={
            "torch_dtype":torch.bfloat16,
        }
    )

    def _call(self, prompt:str, stop=None) -> str:
        response = self.pipeline(prompt,max_new_tokens = max_tokens)[0]["generated_text"]
        return response[len(prompt):]
    
    @property
    def _identifying_params(self):
        return {
            "name_of_model": self.model_name
        }
    
    @property
    def _llm_type(self) -> str:
        return "custom"


@timeit()
def create_index():
    """
    responsible for creating an index
    """
    print("Creating Index")
    # wrapper around the LLMChain for langchain
    llm = LLMPredictor(llm = LocalOPT())
    service_context = ServiceContext.from_defaults(
        llm_predictor = llm,
        prompt_helper = prompt_healper
    )
    docs = SimpleDirectoryReader('news').load_data()
    index = GPTListIndex.from_documents(
        docs,
        service_context = service_context
    )
    print("Done, creating index")
    return index

@timeit()
def execute_query():
    # response = index.query(
    #     "who does indonésia exports its coal to ?",
    #     exclude_keywords = ["petroleum"],
    #     response_mode = "no_text")
    response = index.query(
        "Summarize Australia's coal exports in 2023",
        response_mode = "tree_summarize")
    return response



if __name__ == "__main__":

    filename:str = "demo9.json"
    if not os.path.exists(filename):
        print("No local cache of the model")
        print("Downloading from huggingface... ")
        index = create_index()
        index.save_to_disk(filename)
    else:
        print("Loading Local Cache")
        index = GPTListIndex.load_from_disk(filename)

    response = execute_query()
    print(response)
    print(response.source_nodes())
    print(response.get_formatted_sources())

