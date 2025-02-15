import os
import json
import streamlit as st
# ----------------------------- STREAMLIT INTERFACE ----------------------------- #
# Title and description for the Streamlit app
st.title("Multi-Agent System 🚀")
st.markdown("### Ask me about:")
st.markdown("""
- General queries  
- Math problems  
- Fun facts  
""")
from langchain.agents import Tool, initialize_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
import random

# Set environment variables for authentication  
os.environ["GOOGLE_API_KEY"] = "***************************************"  # Set API key for Google's generative model authentication

# Set up the underlying language model (LLM) with the Gemini model
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-002")

# ----------------------------- AGENT 1: General Queries ----------------------------- #
# This agent handles general user queries using Google's Gemini-1.5-flash-002 model
def general(input=""):
    try:
        # Initialize Google's generative model and generate content based on user input
        results = model.invoke(input)
        print(results.content)
        #results = json.loads(results)  # Convert JSON string to Python dictionary
        return results.content
    except Exception as e:
        # Return the existing results or handle any exceptions
        return results.content

# Define the general tool with its name, function, and description
general_tool = Tool(
    name="General Agent",
    func=general,
    description="Handles general queries using Google's Gemini model."
)

# ----------------------------- AGENT 2: Math Operations ----------------------------- #
# This agent performs mathematical operations by evaluating expressions from user input
def math_tool(input_text):
    try:
        result = eval(input_text)  # Evaluate the user-provided math expression
        return f"The result of {input_text} is {result}"
    except Exception as e:
        # Print error for debugging and return the exception message
        print(f"Error in math tool: {e}")
        return str(e)

# Define the math tool with its name, function, and description
math_agent = Tool(
    name="Math Agent",
    func=math_tool,
    description="Performs mathematical operations such as solving equations and checking prime numbers."
)

# ----------------------------- AGENT 3: Trivia Agent ----------------------------- #
# This agent returns random fun facts from a predefined list
def facts_tool(input_text):
    trivia_facts = [
        "Honey never spoils. Archaeologists have found pots of honey in ancient Egyptian tombs that are over 3000 years old!",
        "Octopuses have three hearts and blue blood.",
        "Bananas are berries, but strawberries aren’t.",
        "Sharks existed before trees—they’ve been around for over 400 million years.",
        "There are more stars in the universe than grains of sand on all the Earth’s beaches."
    ]
    return random.choice(trivia_facts)  # Return a random fun fact

# Define the facts tool with its name, function, and description
trivia_agent = Tool(
    name="Facts Agent",
    func=facts_tool,
    description="Provides random fun facts and trivia."
)

# ----------------------------- INITIALIZE CONVERSATIONAL AGENT ----------------------------- #


# Create a conversation memory buffer to maintain the last 3 messages in the chat history
memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=3,  # Number of past exchanges to remember
    return_messages=True
)

# Initialize the conversational agent with all tools and memory
conversational_agent = initialize_agent(
    agent='chat-conversational-react-description',  # Type of agent
    tools=[general_tool, math_agent, trivia_agent],  # List of tools (agents)
    llm=model,  # Underlying LLM model
    verbose=True,  # Print detailed logs during execution
    max_iterations=3,  # Maximum number of iterations
    early_stopping_method='generate',  # Stop early if output is generated
    memory=memory  # Use conversation memory
)

# ----------------------------- CHAT HISTORY MANAGEMENT ----------------------------- #
# Initialize chat history in session state if not already present
if 'messages' not in st.session_state:
    st.session_state.messages = []

# ----------------------------- USER INPUT ----------------------------- #
# Text input field for user to ask questions
user_input = st.text_input("You: ", "")

if user_input:
    # Add the user message to chat history with an emoji for a better experience
    st.session_state.messages.append(f"🧑‍💻 You: {user_input}")
    
    # Get the response from the conversational agent
    response = conversational_agent(user_input)
    
    # Add the bot's response to the chat history with a bot emoji
    st.session_state.messages.append(f"🤖 Bot: {response['output']}")

# ----------------------------- DISPLAY CHAT HISTORY ----------------------------- #
# Display the full chat history line by line
for message in st.session_state.messages:
    st.write(message)
