import streamlit as st
from langchain.llms.replicate import Replicate
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import LLMChain
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
# from dotenv import load_dotenv
# import os

# Load environment variables
# load_dotenv()            
REPLICATE_API_TOKEN = st.secrets('REPLICATE_API_TOKEN')

# Streamlit app
st.sidebar.title('Models and Parameters')

llm_model ="meta/llama-2-7b-chat:13c3cdee13ee059ab779f0291d29054dab00a47dad8261375654de5540165fb0"

temperature = st.sidebar.slider('Temperature', min_value=0.01, max_value=5.0, value=0.1, step=0.01)
top_p = st.sidebar.slider('Top P', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
max_length = st.sidebar.slider('Max Length', min_value=32, max_value=500, value=120, step=8)

st.title('🦙💬 Llama 2 Chatbot')

# Use st.session_state to maintain the state of chat_history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

llm = Replicate(
    streaming=False,
    model=llm_model,
    model_kwargs={"temperature": temperature, "max_length": max_length, "replicate_api_token": REPLICATE_API_TOKEN},
)

prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            "Answer the following question: {question}"
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{question},")
    ]
)

memory = ConversationBufferWindowMemory(
    memory_key="chat_history", return_messages=True, k=2, input_key="question"
)

conversation = LLMChain(llm=llm, prompt=prompt, verbose=True, memory=memory)

user_query = st.text_input('You:', '')

if st.button('Send'):
    if user_query:
        user_response = {"role": "user", "content": user_query}
        st.session_state.chat_history.append(user_response)

        response = conversation({"question": user_query})
        bot_response = response["text"]
        bot_response = {"role": "bot", "content": bot_response}
        st.session_state.chat_history.append(bot_response)

# Display chat history using st.session_state
st.subheader('Chat History')
history_text = ""
for message in st.session_state.chat_history:
    if message["role"] == "user":
        history_text += f'You: {message["content"]}\n'
    else:
        history_text += f'Bot: {message["content"]}\n'
st.text_area("Conversation History", value=history_text, height=400)
