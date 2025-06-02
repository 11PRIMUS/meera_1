import os
import json
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
import streamlit as st

USER_DATA_DIR = Path("user_data")
USER_DATA_DIR.mkdir(exist_ok=True) 


def get_user_chat_history_path(username: str) -> Path:
    return USER_DATA_DIR / f"{username}_chat_history.json"

def load_user_data(username: str) -> tuple[list, ChatMessageHistory]:
    messages_display = []
    chat_history_store = ChatMessageHistory()
    history_file = get_user_chat_history_path(username)
    if history_file.exists():
        try:
            with open(history_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                messages_display = data.get("messages_display", [])
                # reconstruct ChatMessageHistory
                stored_messages = data.get("chat_history_store", [])
                for msg_data in stored_messages:
                    if msg_data.get("type") == "human":
                        chat_history_store.add_user_message(msg_data["content"])
                    elif msg_data.get("type") == "ai":
                        chat_history_store.add_ai_message(msg_data["content"])
        except json.JSONDecodeError:
            st.warning(f"failed to get history file for {username} starting with a fresh history.")
        except Exception as e:
            st.error(f"error {username}: {e}. ")
            
    return messages_display, chat_history_store

def save_user_data(username: str, messages_display: list, chat_history_store: ChatMessageHistory):
    history_file = get_user_chat_history_path(username)
    
    serializable_chat_history = []
    for msg in chat_history_store.messages:
        serializable_chat_history.append({"type": msg.type, "content": msg.content})
    
    try:
        with open(history_file, "w", encoding="utf-8") as f:
            json.dump({
                "messages_display": messages_display,
                "chat_history_store": serializable_chat_history
            }, f, indent=2)
    except Exception as e:
        st.error(f"Error saving history for {username}: {e}")

os.environ["LANGCHAIN_TRACING_V2"]="true"
langchain_api_key_from_env=st.secrets.get("LANGCHAIN_API_KEY")
if langchain_api_key_from_env:
    os.environ["LANGCHAIN_API_KEY"] = langchain_api_key_from_env
else:
    st.sidebar.warning("LANGCHAIN_API_KEY not found. LangSmith tracing disabled")

#user name
st.sidebar.title("User Login")
if "username" not in st.session_state:
    st.session_state.username = ""

username_input = st.sidebar.text_input("Enter your username:", value=st.session_state.username, key="username_field")

if username_input != st.session_state.username:

    if st.session_state.username: # if there was a previous username
        old_chat_history_key = f"{st.session_state.username}_chat_history_store"
        old_messages_display_key = f"{st.session_state.username}_messages_display"
        if old_chat_history_key in st.session_state:
            del st.session_state[old_chat_history_key]
        if old_messages_display_key in st.session_state:
            del st.session_state[old_messages_display_key]
    st.session_state.username = username_input
    if st.session_state.username: #rerun if a new username is actually entered
        st.rerun()


username = st.session_state.username

if not username:
    st.info("Please enter a username in the sidebar to begin.")
    st.stop()

#usre session state keys
CHAT_HISTORY_STORE_KEY = f"{username}_chat_history_store"
MESSAGES_DISPLAY_KEY = f"{username}_messages_display"

#session state for the current user
if CHAT_HISTORY_STORE_KEY not in st.session_state or MESSAGES_DISPLAY_KEY not in st.session_state :
    messages_display_loaded, chat_history_store_loaded = load_user_data(username)
    st.session_state[MESSAGES_DISPLAY_KEY] = messages_display_loaded
    st.session_state[CHAT_HISTORY_STORE_KEY] = chat_history_store_loaded


prompt = ChatPromptTemplate.from_messages([
    ("system", "You are Meera, an emotional assistant. You reflect on your conversations and keep a sort of internal diary to help you remember and understand the user better over time. Respond to user queries, drawing upon your understanding from past interactions and your internal reflections."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{question}")
]) 

st.title(f'EMO with custom model (User: {username})') 

llm=None 
NEBIUS_API_KEY=st.secrets.get("NEBIUS_API_KEY")
NEBIUS_BASE_URL="https://api.studio.nebius.com/v1/" 
NEBIUS_MODEL_NAME="Qwen/Qwen3-4B-fast-LoRa:meera4b-WqNr" 

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
        st.error(f"failed to init model with Nebius: {e}")
        llm = None

output_parser=StrOutputParser()


for msg_idx, msg in enumerate(st.session_state.get(MESSAGES_DISPLAY_KEY, [])):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

#new user input
if new_user_input := st.chat_input(f"Tell Meera about your day, {username}..."):
    st.session_state[MESSAGES_DISPLAY_KEY].append({"role":"user","content":new_user_input})
    with st.chat_message("user"):
        st.markdown(new_user_input)

    if llm: 
        #chat history for the current user
        loaded_chat_history=st.session_state[CHAT_HISTORY_STORE_KEY].messages
        
        chain=prompt|llm|output_parser
        try:
            response=chain.invoke({
                "question":new_user_input,
                "chat_history":loaded_chat_history
            })
            st.session_state[MESSAGES_DISPLAY_KEY].append({"role":"assistant","content":response})
            
            with st.chat_message("assistant"):
                st.markdown(response)

            #save interaction for the current user
            st.session_state[CHAT_HISTORY_STORE_KEY].add_user_message(new_user_input)
            st.session_state[CHAT_HISTORY_STORE_KEY].add_ai_message(response)
            save_user_data(username, st.session_state[MESSAGES_DISPLAY_KEY], st.session_state[CHAT_HISTORY_STORE_KEY])
            st.rerun() 

        except Exception as e:
            error_message = f"error during model chain invocation: {e}"
            st.error(error_message)
            st.session_state[MESSAGES_DISPLAY_KEY].append({"role":"assistant","content":f"Sorry, I encountered an issue: {e}"})
            with st.chat_message("assistant"):
                st.markdown(f"Sorry, I encountered an issue: {e}")
                
    elif not missing_configs: #model init failed
        warning_message = "Model not initialized. Please check configurations." 
        st.warning(warning_message)
        st.session_state[MESSAGES_DISPLAY_KEY].append({"role":"assistant","content": warning_message})
        with st.chat_message("assistant"):
            st.markdown(warning_message)