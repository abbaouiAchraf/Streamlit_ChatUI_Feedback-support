import csv
import os
from datetime import datetime
import streamlit as st
from langchain_community.llms.openai import OpenAIChat
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.callbacks.tracers.run_collector import RunCollectorCallbackHandler
from langchain.memory import ConversationBufferMemory, StreamlitChatMessageHistory
from langchain.schema.runnable import RunnableConfig
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from streamlit_feedback import streamlit_feedback

# CONFIGURATION

st.set_page_config(
    page_title="Chat with the Streamlit",
    page_icon="🤖",
)

os.environ["OPENAI_API_KEY"] = "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"
openai_ = OpenAIChat()

# Define the CSV file path
csv_file_path = "feedback_data.csv"
field_names = ["Query", "Answer", "Feedback", "Timestamp"]

feedback_option = "faces"  if st.toggle(label="`Thumbs` ⇄ `Faces`", value=False) else "thumbs"

MAX_CHAR_LIMIT = 500

system_prompt = "" # To add costume system prompt to determine the ai behavior
system_prompt = system_prompt.strip().replace("{", "{{").replace("}", "}}")

# UTILITY FUNCTIONS

def append_to_csv(query, answer, feedback):
    """
    Appends a row to a CSV file with the provided query, answer, feedback, and timestamp.

    Parameters:
        query (str): The user query.
        answer (str): The generated answer/response.
        feedback (str): Feedback related to the response.

    Returns:
        None
    """
    file_empty = os.stat(csv_file_path).st_size == 0

    with open(csv_file_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=field_names)

        if file_empty:
            writer.writeheader()

        writer.writerow({"Query": query, "Answer": answer, "Feedback": feedback, "Timestamp": datetime.now()})


def push_to_s3_bucket(file_path, bucket_name):
    """
    Pushes data to an Amazon S3 bucket.

    Parameters:
        file_path (str): The  path of the csv file.
        bucket_name (str): The name of the bucket to push the csv file in it.

    Returns:
        None
    """
    pass


def get_llm_chain(system_prompt: str, memory: ConversationBufferMemory) -> LLMChain:
    """
    Initializes and returns an instance of LLMChain.

    Parameters:
        system_prompt (str): The system prompt to start the conversation.
        memory (ConversationBufferMemory): Memory for conversation history.

    Returns:
        LLMChain: An instance of LLMChain.
    """
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt + "\nIt's currently {time}."),
         MessagesPlaceholder(variable_name="chat_history"),
         ("human", "{input}")]).partial(time=lambda: str(datetime.now()))
    llm = ChatOpenAI(temperature=0.7)
    chain = LLMChain(prompt=prompt, llm=llm, memory=memory)
    return chain


def _get_openai_type(msg):
    """
        Returns the type of the message (user, assistant, or other).

        Parameters:
            msg: The message object.

        Returns:
            str: The type of the message.
    """
    if msg.type == "human":
        return "user"
    if msg.type == "ai":
        return "assistant"
    if msg.type == "chat":
        return msg.role
    return msg.type


# MAIN

if "run_id" not in st.session_state:
    st.session_state.run_id = None

memory = ConversationBufferMemory(
    chat_memory=StreamlitChatMessageHistory(key="langchain_messages"),
    return_messages=True,
    memory_key="chat_history",
)

chain = get_llm_chain(system_prompt, memory)

if st.sidebar.button("Clear message history"):
    memory.clear()
    st.session_state.run_id = None

for msg in st.session_state.langchain_messages:
    streamlit_type = _get_openai_type(msg)
    avatar = "🤖" if streamlit_type == "assistant" else None
    with st.chat_message(streamlit_type, avatar=avatar):
        st.markdown(msg.content)

run_collector = RunCollectorCallbackHandler()
runnable_config = RunnableConfig(
    callbacks=[run_collector],
    tags=["Streamlit Chat"],
)

if prompt := st.chat_input(placeholder="Ask anything you want!"):
    if len(prompt) > MAX_CHAR_LIMIT:
        st.warning(f"⚠️ Your input is too long! Please limit your input to {MAX_CHAR_LIMIT} characters.")
        prompt = None
    else:
        st.chat_message("user").write(prompt)
        # _reset_feedback()
        with st.chat_message("assistant", avatar="🤖"):
            message_placeholder = st.empty()
            full_response: str
            input_structure = {"input": prompt}
            message_placeholder.markdown("Hmm...")
            full_response = chain.invoke(input_structure, config=runnable_config)["text"]
            message_placeholder.markdown(full_response)
            run = run_collector.traced_runs[0]
            run_collector.traced_runs = []
            st.session_state.run_id = run.id

            # store prompt and full_response in a session variable, to not lose them
            st.session_state.prompt = prompt
            st.session_state.full_response = full_response

if st.session_state.get("run_id"):
    feedback = streamlit_feedback(
        feedback_type=feedback_option,
        optional_text_label="[Optional] Please provide an explanation",
        key=f"feedback_{st.session_state.run_id}",
    )

    score_mappings = {
        "thumbs": {"👍": 1, "👎": 0},
        "faces": {"😀": 1, "🙂": 0.75, "😐": 0.5, "🙁": 0.25, "😞": 0},
    }

    scores = score_mappings[feedback_option]

    if feedback:
        score = scores.get(feedback["score"])
        if score is not None:
            feedback_type_str = f"score: score - {feedback['score']}"
            append_to_csv(st.session_state.prompt, st.session_state.full_response, feedback_type_str)
            # add to S3 bucket code here
            st.success("Feedback saved successfully!")
        else:
            st.warning("Invalid feedback score.")
