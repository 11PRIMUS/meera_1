import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"]="true"
langchain_api_key_from_env = os.getenv("LANGCHAIN_API_KEY")
if langchain_api_key_from_env:
    os.environ["LANGCHAIN_API_KEY"] = langchain_api_key_from_env
else:
    st.sidebar.warning("check your langchain api key")

prompt=ChatPromptTemplate.from_messages(
    [
    ("system","You are Meera a emotional assistant.Respond to user queries"),
    ("user","Question:{question}")
    ]
)
st.title(' EMO with custom nebius api')
input_text=st.text_input("tell about your day or mood")

llm=None 
NEBIUS_API_KEY=os.getenv("NEBIUS_API_KEY")
NEBIUS_BASE_URL="https://api.studio.nebius.com/v1/"
NEBIUS_MODEL_NAME="meta-llama/Llama-3.2-3B-Instruct-LoRa:emo-Pfnh" 

missing_configs = []
if not NEBIUS_API_KEY:
    missing_configs.append("nebius api keymust be in .env")
if not NEBIUS_BASE_URL:
    missing_configs.append("check base url")
if not NEBIUS_MODEL_NAME:
    missing_configs.append("check model name")

if missing_configs:
    st.error(f"Nebius configuration is incomplete. Missing: {', '.join(missing_configs)}.")
else:
    try:
        llm = ChatOpenAI( 
                model=NEBIUS_MODEL_NAME,
                api_key=NEBIUS_API_KEY,    
                base_url=NEBIUS_BASE_URL  
            )
    except Exception as e:
        st.error(f"Failed to initialize LLM with Nebius: {e}")
        llm = None

output_parser=StrOutputParser()

if input_text: 
    if llm: 
        chain=prompt|llm|output_parser
        try:
            response = chain.invoke({'question':input_text})
            st.write(response)
        except Exception as e:
            st.error(f"Error during LLM chain invocation: {e}")
    elif not missing_configs: 
        st.warning("LLM is not available. Initialization failed (see error above if any).")
elif not input_text and llm:
    st.info("Enter your mood or day to get a response.")

