
import streamlit as st
import json
from langchain_groq import ChatGroq
from langchain import PromptTemplate
from langchain.chains import LLMChain
from kor.extraction import create_extraction_chain
from kor import Object, Text
import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Retrieve the API key
api_key = os.getenv("API_KEY")
 
# Set the GROQ API key securely
os.environ["GROQ_API_KEY"] = api_key

# Configure Streamlit page
st.set_page_config(page_title="Medical Chatbot", layout="wide")
st.markdown(
    """
    <style>
        body {
            background-color: black;
            color: white;
        }
        .chat-container {
            display: flex;
            flex-direction: row;
        }
        .left-box {
            width: 60%;
            padding-right: 15px;
        }
        .right-box {
            width: 40%;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize the ChatGroq model
model = ChatGroq(model="llama3-8b-8192", temperature=0.5)

from kor import Object, Text

# Schema to classify questions based on category and intention
schema = Object(
    id="question_analysis",
    attributes=[
        Text(
            id="category",
            description="Category of the question (e.g.,medical, non-relevant)",
            examples=[
                ("can i do 10km marathon?", "medical"),
                ("Do i need to worry about cancer?", "medical"),
                ("chennai red alert", "non-relevant"),
            ],
        ),
        Text(
            id="intention",
            description="User's intention or query type (e.g., fitness inquiry, health concern, common inquiries)",
            examples=[
                ("can i do 10km marathon?", "fitness inquiry"),
                ("latest news on tesla", "non-relevant inquiry"),
            ],
        ),
    ],
    examples=[
        (
            "Can I do 20km marathon?",
            [
                {"category": "medical", "intention": "fitness inquiry"},
            ],
        )
    ],
    many=True,
)
chain1 = create_extraction_chain(model, schema)

template = """
You are a chatbot only answers medical related questions of your own report. Reply based on the Category&Intention ignore non-relevant and common queries politely and shortly. 
Do not mention 'JSON data' or provide unnecessary explanations. 
Start directly with the necessary information related to the query. 


JSON Data: {json_data}

Question: {question}

Category&Intention: {catogory_intention}

Answer:
"""

prompt = PromptTemplate(
    input_variables=["json_data", "question", "catogory_intention"],  # Ensure only these keys are used
    template=template
)
# Build the LLM chain
chain = LLMChain(llm=model, prompt=prompt)

# Define the function to interact with the chatbot
def query_chatbot(data,question):
    json_data_str = json.dumps(data, indent=2)  # Convert JSON to string for the prompt
    cat_inten = chain1.invoke(question)
    response = chain.run(json_data=json_data_str, question=question, catogory_intention = cat_inten)
    return response


# Streamlit app logic
st.title("Medical Chatbot")
st.write("Upload your JSON file and enter a query to interact with the medical chatbot.")

# Layout with two columns (chat interface)
col1, col2 = st.columns([3, 2])

# Right-side input box: Upload JSON file and ask questions
with col2:
    uploaded_file = st.file_uploader("Upload JSON file", type=["json"])
    if uploaded_file is not None:
        data = json.load(uploaded_file)  # Load the JSON file
        question = st.text_input("Enter your query")  # Input query box
        if st.button("Ask"):
            answer = query_chatbot(data, question)
            st.session_state.chat_history.append((question, answer))

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Left-side chat display
with col1:
    st.subheader("Chat History")
    for question, answer in st.session_state.chat_history:
        st.write(f"**You:** {question}")
        st.write(f"**Chatbot:** {answer}")
