import os
import json 
from pathlib import Path 
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage 
import streamlit as st



os.environ["LANGCHAIN_TRACING_V2"]="true"
langchain_api_key_from_env=st.secrets.get("LANGCHAIN_API_KEY")
if langchain_api_key_from_env:
    os.environ["LANGCHAIN_API_KEY"] = langchain_api_key_from_env
else:
    st.sidebar.warning("LANGCHAIN_API_KEY not found. LangSmith tracing disabled")

#session state
if "chat_history_store" not in st.session_state:
    st.session_state.chat_history_store = ChatMessageHistory()

if "messages_display" not in st.session_state:
    st.session_state.messages_display=[]

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are Meera, an emotional assistant. You are new to understanding human emotions and interactions, but you learn from every conversation.You reflect on your conversations and keep a sort of internal diary to help you remember and understand the user better over time. Respond to user queries, drawing upon your understanding from past interactions and your internal reflections. If you don't have much past interaction to draw from, your responses might seem more naive or basic, which is okay as you are learning."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{question}")
]) 

st.title(f'EMO with custom model (User: {username})') 

llm=None 
NEBIUS_API_KEY=st.secrets.get("NEBIUS_API_KEY")
NEBIUS_BASE_URL="https://api.studio.nebius.com/v1/" 
NEBIUS_MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct-LoRa:meera2-PJwE" 

missing_configs = []
if not NEBIUS_API_KEY:
    missing_configs.append("NEBIUS_API_KEY")
if not NEBIUS_BASE_URL:
    missing_configs.append("Nebius API Base URL is missing") 
if not NEBIUS_MODEL_NAME:
    missing_configs.append("Nebius Model Name is missing") 

if missing_configs:
    st.error(f"Nebius configuration Missing: {', '.join(missing_configs)}.")
else:
    try:
        llm = ChatOpenAI( 
                model=NEBIUS_MODEL_NAME,
                api_key=NEBIUS_API_KEY,    
                base_url=NEBIUS_BASE_URL  
            )
    except Exception as e:
        st.error(f"filed to init model with Nebius: {e}")
        llm = None

output_parser=StrOutputParser()

#chat messages for the current user
for msg_idx, msg in enumerate(st.session_state.get(MESSAGES_DISPLAY_KEY, [])):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # The "hear meera voice" button and its logic are removed.

#new user input
if new_user_input := st.chat_input(f"Tell Meera about your day, {username}..."):
    st.session_state[MESSAGES_DISPLAY_KEY].append({"role":"user","content":new_user_input})
    with st.chat_message("user"):
        st.markdown(new_user_input)

    if llm: 
        #chat history
        loaded_chat_history=st.session_state.chat_history_store.messages
        
        chain=prompt|llm|output_parser
        try:
            response=chain.invoke({
                "question":new_user_input,
                "chat_history":final_chat_history_for_llm #combined history
            })
            st.session_state[MESSAGES_DISPLAY_KEY].append({"role":"assistant","content":response})
            
            with st.chat_message("assistant"):
                st.markdown(response)

            #save interaction
            st.session_state.chat_history_store.add_user_message(new_user_input)
            st.session_state.chat_history_store.add_ai_message(response)
            st.rerun() 

        except Exception as e:
            error_message = f"error during model chain invocation: {e}"
            st.error(error_message)
            st.session_state[MESSAGES_DISPLAY_KEY].append({"role":"assistant","content":f"Sorry, I encountered an issue: {e}"})
            with st.chat_message("assistant"):
                st.markdown(f"sorry, I encountered an issue: {e}")
                
    elif not missing_configs: #model init failed
        warning_message = "check model " # Consider a more descriptive message
        st.warning(warning_message)
        st.session_state[MESSAGES_DISPLAY_KEY].append({"role":"assistant","content": warning_message})
        with st.chat_message("assistant"):
            st.markdown(warning_message)
    elif missing_configs:
        st.warning(f"nebius missing config{', '.join(missing_configs)}")

