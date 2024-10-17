# Langchain
LangChain is an advanced framework designed to streamline the development of applications powered by language models. It provides essential tools for building chatbots, question-answering systems, and generative AI applications by allowing seamless integration with large language models (LLMs). With LangChain, developers can efficiently manage workflows like prompt engineering, memory handling, document loading, and retrieval-based queries.

# Kor 
Kor complements LangChain by offering a structured way to define and extract schemas from text. It simplifies the interaction between unstructured data and LLMs, allowing the creation of structured responses using components like Text, Array, and Object. This is particularly useful for applications such as medical assistants, where extracting and organizing critical information accurately is essential for reliable performance.

# Medical Assistant Chatbot: Powered by LangChain and Kor
## Initialize an LLM 
Initializes the GROQ API using an environment-stored key and sets up the ChatGroq model ("llama3-8b-8192") with a temperature of 0.5 for balanced responses.
```python
api_key = os.getenv("API_KEY")
os.environ["GROQ_API_KEY"] = api_key
model = ChatGroq(model="llama3-8b-8192", temperature=0.5)
```
## Define schema
Define a schema for analyzing user questions by categorizing them and identifying their intentions. Using the Object function, it creates a schema called "question_analysis" that consists of two text attributes: "category" and "intention." The "category" attribute classifies questions into types such as "medical" or "non-relevant," while the "intention" attribute identifies the user's query type, like "fitness inquiry" or "non-relevant inquiry.
```python
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
```

## Prompt template 
Define a prompt template for a medical chatbot that provides concise answers based on "Category&Intention" while ignoring non-relevant questions. It instructs the chatbot to avoid mentioning "JSON data" and unnecessary explanations, ensuring direct responses. A PromptTemplate is defined with specific input variables, and an LLM chain is established to enable the chatbot to generate relevant replies
```python
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
```
## Function for Interacting with Medical Chatbot
This function formats the input data into a readable string and determines the "Category&Intention" of the user's question using an extraction chain. The function then calls the language model chain with the formatted data, user question, and extracted information, ultimately returning the chatbot's response that provides relevant medical insights.
```python
def query_chatbot(data,question):
    json_data_str = json.dumps(data, indent=2)  # Convert JSON to string for the prompt
    cat_inten = chain1.invoke(question)
    response = chain.run(json_data=json_data_str, question=question, catogory_intention = cat_inten)
    return response
```
## Explore the Streamlit app here:
https://medical-report-assistant-z6wvctwe6akuy5hw8vcrsc.streamlit.app/
