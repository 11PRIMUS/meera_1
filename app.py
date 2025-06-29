import os
import json
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import sqlite3
import datetime
import requests
import base64
import streamlit.components.v1 as components
import io
import soundfile as sf
from veena_tts import generate_speech

DB_NAME="meera_chat.db"

def init_db():
    conn=sqlite3.connect(DB_NAME)
    cursor=conn.cursor()
    cursor.execute("""CREATE TABLE IF NOT EXISTS chat_messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL,
        message_type TEXT NOT NULL,
        content TEXT NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)
    """)
    conn.commit()
    conn.close()

init_db()

def load_user_data(username: str) -> tuple[list, ChatMessageHistory]:
    messages_display = []
    chat_history_store = ChatMessageHistory()
    conn=sqlite3.connect(DB_NAME)
    cursor=conn.cursor()
    cursor.execute("select msg_tye,content from chat_msg where username=? order by timestamp asc",(username,))
    for row in cursor.fetchall():
        msg_type, content = row
        role = "user" if msg_type == "human" else "assistant"
        messages_display.append({"role": role, "content": content})
        if msg_type == "human":
            chat_history_store.add_user_message(content)
        elif msg_type == "ai":
            chat_history_store.add_ai_message(content)
    conn.close()
    return messages_display, chat_history_store

def save_message_to_db(username: str, message_type: str, content: str):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO chat_messages (username, message_type, content, timestamp) VALUES (?, ?, ?, ?)",
        (username, message_type, content, datetime.datetime.now())
    )
    conn.commit()
    conn.close()

config_file_path = Path(__file__).parent / "config.yaml"
if not config_file_path.exists():
    st.error("config.yaml not found")
    st.stop()

with open(config_file_path) as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    #config['preauthorized']
)

os.environ["LANGCHAIN_TRACING_V2"]="true"
langchain_api_key_from_env=st.secrets.get("LANGCHAIN_API_KEY")
if langchain_api_key_from_env:
    os.environ["LANGCHAIN_API_KEY"] = langchain_api_key_from_env

name, authentication_status, username = authenticator.login()

if authentication_status == False:
    st.error("Username/password is incorrect")
    st.stop()
elif authentication_status == None:
    st.warning(" your username and password")
    if not langchain_api_key_from_env:
        st.sidebar.warning("LANGCHAIN_API_KEY not found")
    st.stop()

st.sidebar.success(f"Welcome *{name}*")
authenticator.logout("Logout", "sidebar")

if not langchain_api_key_from_env:
    st.sidebar.warning("tracing disabled")

CHAT_HISTORY_STORE_KEY = f"{username}_chat_history_store"
MESSAGES_DISPLAY_KEY = f"{username}_messages_display"

if CHAT_HISTORY_STORE_KEY not in st.session_state or MESSAGES_DISPLAY_KEY not in st.session_state:
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
NEBIUS_MODEL_NAME=""

missing_configs = []
if not NEBIUS_API_KEY:
    missing_configs.append("NEBIUS_API_KEY")
if not NEBIUS_BASE_URL:
    missing_configs.append("base url missing")
if not NEBIUS_MODEL_NAME:
    missing_configs.append("base model Name is missing")

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
    if msg["role"] == "assistant":
        with st.chat_message("assistant"):
            st.markdown(msg["content"])
            if st.button("talk To", key=f"talk_{msg_idx}"):
                st.session_state["speak_text"] = msg["content"]
    else:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

if "speak_text" in st.session_state:
    text_to_speak = st.session_state.pop("speak_text")
    try:
        #veena voice
        audio = generate_speech(text_to_speak, speaker="kavya") 
        #buffer write
        buf = io.BytesIO()
        sf.write(buf, audio, 24000, format="WAV")
        st.audio(buf.getvalue(), format="audio/wav")
    except Exception as e:
        st.error(f"Error generating audio: {e}")

if new_user_input := st.chat_input(f"Tell Meera about your day, {name}..."):
    st.session_state[MESSAGES_DISPLAY_KEY].append({"role":"user","content":new_user_input})
    st.session_state[CHAT_HISTORY_STORE_KEY].add_user_message(new_user_input)
    save_message_to_db(username,"human",new_user_input)
    with st.chat_message("user"):
        st.markdown(new_user_input)

    if llm:
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
            st.session_state[CHAT_HISTORY_STORE_KEY].add_user_message(new_user_input)
            st.session_state[CHAT_HISTORY_STORE_KEY].add_ai_message(response)
            save_message_to_db(username, "ai",response)
        except Exception as e:
            error_message = f"error during model chain invocation: {e}"
            st.error(error_message)
            error_response_content=f"sorry, i encountered an issue: {e}"
            st.session_state[MESSAGES_DISPLAY_KEY].append({"role":"assistant","content":error_response_content})
            with st.chat_message("assistant"):
                st.markdown(f"Sorry, I encountered an issue: {e}")
    elif not missing_configs:
        warning_message = "Model not initialized. Please check configurations."
        st.warning(warning_message)
        st.session_state[MESSAGES_DISPLAY_KEY].append({"role":"assistant","content": warning_message})
        with st.chat_message("assistant"):
            st.markdown(warning_message)

    st.rerun()
