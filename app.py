import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
import streamlit as st
from datetime import datetime
from supermemory import supermemory


os.environ["LANGCHAIN_TRACING_V2"]="true"
langchain_api_key_from_env=st.secrets.get("LANGCHAIN_API_KEY")
if langchain_api_key_from_env:
    os.environ["LANGCHAIN_API_KEY"] = langchain_api_key_from_env
else:
    st.sidebar.warning("LANGCHAIN_API_KEY not found. LangSmith tracing disabled")

if "supermemory_client" not in st.session_state:
    st.session_state.supermemory_client=None
    st.session_state.supermemory_enabled=False
    if supermemory:
        SUPERMEMORY_API_KEY=st.secrets.get("SUPERMEMORY_API_KEY")
        if SUPERMEMORY_API_KEY:
            try:
                st.session_state.supermemory_client=supermemory(api_key=SUPERMEMORY_API_KEY)
                st.session_state.supermemory_enabled=True
                st.sidebar.success("sm init")
            except Exception as e:
                st.sidebar.error(f"failed to get sm {e}")
        else:
            st.sidebar.warning("check sm key")
    else:
        st.sidebar.warning("sm lib ")


#session state for chat
if "chat_history_store" not in st.session_state:
    st.session_state.chat_history_store=ChatMessageHistory()

if "messages_display" not in st.session_state:
    st.session_state.messages_display=[]

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are Meera, an emotional assistant. You are new to understanding human emotions and interactions, but you learn from every conversation.You reflect on your conversations and keep a sort of internal diary to help you remember and understand the user better over time. Respond to user queries, drawing upon your understanding from past interactions and your internal reflections. If you don't have much past interaction to draw from, your responses might seem more naive or basic, which is okay as you are learning."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{question}")
]) 

st.title('EMO with custom model ') 

llm=None 
NEBIUS_API_KEY=st.secrets.get("NEBIUS_API_KEY")
NEBIUS_BASE_URL="https://api.studio.nebius.com/v1/" 
NEBIUS_MODEL_NAME="meta-llama/Llama-3.2-3B-Instruct-LoRa:emo-Pfnh" 

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
        llm=None

output_parser=StrOutputParser()

for msg_idx, msg in enumerate(st.session_state.messages_display):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        #"hear meera voice" button is removed temporarily

#new user input
if new_user_input := st.chat_input("tell meera about your day or mood..."):
    st.session_state.messages_display.append({"role":"user","content":new_user_input})
    with st.chat_message("user"):
        st.markdown(new_user_input)

    if llm: 
        #sm adding
        final_chat_history_for_llm = []

        if st.session_state.supermemory_enabled and st.session_state.supermemory_client:
            try:
                query_for_memory=new_user_input 
                search_response=st.session_state.supermemory_client.search.execute(q=query_for_memory, top_k=3)
                
                retrieved_memories=[]
                if search_response and hasattr(search_response, 'memories') and isinstance(search_response.memories, list):
                    memory_items =search_response.memories
                    for item in memory_items:
                        content=item.get('text') #content
                        role=item.get('metadata', {}).get('role')
                        if content and role:
                            if role=="user":
                                retrieved_memories.append(HumanMessage(content=content))
                            elif role=="assistant" or role=="ai": #ai role
                                retrieved_memories.append(AIMessage(content=content))
                elif isinstance(search_response, list):
                     for item in search_response: #assuming item is a dict with 'text' and 'metadata'
                        content=item.get('text')
                        role=item.get('metadata', {}).get('role')
                        if content and role:
                            if role=="user":
                                retrieved_memories.append(HumanMessage(content=content))
                            elif role=="assistant" or role =="ai":
                                retrieved_memories.append(AIMessage(content=content))

                final_chat_history_for_llm.extend(retrieved_memories)
                if retrieved_memories:
                    st.sidebar.info(f"Retrieved {len(retrieved_memories)} memories from Supermemory.")

            except Exception as e:
                st.sidebar.error(f"Error retrieving from Supermemory: {e}")
        
        #current in session history
        final_chat_history_for_llm.extend(st.session_state.chat_history_store.messages)
        
        chain=prompt|llm|output_parser
        try:
            response=chain.invoke({
                "question":new_user_input,
                "chat_history":final_chat_history_for_llm #combined history
            })
            st.session_state.messages_display.append({"role":"assistant","content":response})
            
            with st.chat_message("assistant"):
                st.markdown(response)

            #save interaction to current session history
            st.session_state.chat_history_store.add_user_message(new_user_input)
            st.session_state.chat_history_store.add_ai_message(response)

            #interaction to sm
            if st.session_state.supermemory_enabled and st.session_state.supermemory_client:
                try:
                    current_time=datetime.now().isoformat() #generic id for now
                    user_session_id = "default_user" 
                    
                    st.session_state.supermemory_client.data.create(
                        content=new_user_input,
                        metadata={"role": "user", "timestamp": current_time, "session_id": user_session_id}
                    )
                    st.session_state.supermemory_client.data.create(
                        content=response,
                        metadata={"role": "assistant", "timestamp": current_time, "session_id": user_session_id}
                    )
                    st.sidebar.info("saved to sm")
                except Exception as e:
                    st.sidebar.error(f"error saving to sm {e}")
            
            st.rerun() 

        except Exception as e:
            error_message = f"error during model chain invocation: {e}"
            st.error(error_message)
            st.session_state.messages_display.append({"role":"assistant","content":f"Sorry, I encountered an issue: {e}"})
            with st.chat_message("assistant"):
                st.markdown(f"sorry, I encountered an issue: {e}")
                
    elif not missing_configs: #model init failed
        warning_message = "check nebius" 
        st.warning(warning_message)
        st.session_state.messages_display.append({"role":"assistant","content": warning_message})
        with st.chat_message("assistant"):
            st.markdown(warning_message)
    elif missing_configs:
        st.warning(f"nebius missing config{', '.join(missing_configs)}")

