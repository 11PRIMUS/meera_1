import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"]="true"
langchain_api_key_from_env = os.getenv("LANGCHAIN_API_KEY")
if langchain_api_key_from_env:
    os.environ["LANGCHAIN_API_KEY"] = langchain_api_key_from_env
else:
    st.sidebar.warning("check your langchain api key")

if "memory" not in st.session_state:
    st.session_state.memory=ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True)
if "messages_display" not in st.session_state:
    st.session_state.messages_display=[]

prompt=ChatPromptTemplate.from_messages(
    [
    ("system","You are Meera a emotional assistant.Respond to user queries"),
    ("user","Question:{question}")
    ]
)
st.title(' EMO with custom nebius api')

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
#existing messages
for msg in st.session_state.messages_display:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

#new input
if new_user_input :=st.chat_input("tell Meera about your day or mood..."):
    st.session_state.messages_display.append({"role":"user","content":new_user_input})
    with st.chat_message("user"):
        st.markdown(new_user_input)

    if llm:
        loaded_chat_history=st.session_state.memory.chat_memory.messages
        chain=prompt|llm|output_parser
        try:
            response = chain.invoke({
                "question": new_user_input,
                "chat_history": loaded_chat_history
            })
            st.session_state.messages_display.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)
        
            st.session_state.memory.save_context(  #save current interaction
                {"input": new_user_input}, 
                {"output": response}
            )
        except Exception as e:
            error_message = f"Error during LLM chain invocation: {e}"
            st.error(error_message)
            st.session_state.messages_display.append({"role": "assistant", "content": f"Sorry, I encountered an issue: {e}"})
            with st.chat_message("assistant"):
                st.markdown(f"Sorry, I encountered an issue: {e}")
                
    elif not missing_configs: 
        warning_message = "LLM is not available. Initialization may have failed."
        st.warning(warning_message)
        st.session_state.messages_display.append({"role": "assistant", "content": warning_message})
        with st.chat_message("assistant"):
            st.markdown(warning_message)
# if input_text: 
#     if llm: 
#         chain=prompt|llm|output_parser
#         try:
#             response = chain.invoke({'question':input_text})
#             st.write(response)
#         except Exception as e:
#             st.error(f"Error during LLM chain invocation: {e}")
#     elif not missing_configs: 
#         st.warning("LLM is not available. Initialization failed (see error above if any).")
# elif not input_text and llm:
#     st.info("Enter your mood or day to get a response.")

