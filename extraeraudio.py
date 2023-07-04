import os
from dotenv import load_dotenv, find_dotenv
from langchain import HuggingFaceHub
from langchain import PromptTemplate, LLMChain, OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import YoutubeLoader
import textwrap

# Load environment variables from .env file
load_dotenv()


HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

repo_id = "tiiuae/falcon-7b"
llm = HuggingFaceHub(huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN, 
                     repo_id=repo_id, 
                     model_kwargs={"temperature":0.7, "max_new_tokens":4000})

video_url = "https://www.youtube.com/watch?v=v2AC41dglnM"
loader = YoutubeLoader.from_youtube_url(video_url, language= "es")
transcript = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 3000)
docs = text_splitter.split_documents(transcript)

chain = load_summarize_chain(llm, chain_type = "map_reduce", verbose = True)
print(chain.llm_chain.prompt.template)
print(chain.combine_document_chain.llm_chain.prompt.template)

output_summary = chain.run(docs)
wrapped_text = textwrap.fill(
    output_summary, width = 100, break_long_words = False, replace_whitespace = False)

print(wrapped_text)