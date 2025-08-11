import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from typing import Dict, List

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
rajat_info="""
I am Rajat Ranvir 21 years old MERN Full stack developer ,Data scientist and AI/ML Enthusiast and i am pursuing career btech engineering in computer science and engineering  in deogiri college of engineering and management studies aurangabad through out this engineering 4 years have learned a rich and diverse technical skill set spanning multiple categories. In Languages, you excel in Python, Java, C, HTML, CSS, and JavaScript. Your Frameworks expertise includes React, Node.js, Express, Django, TailwindCSS, and Gradio. In Libraries, you are skilled in Pandas, Numpy, Matplotlib, Scikit-Learn, Requests, BeautifulSoup, and Uvicorn. Your Databases knowledge covers MongoDB, MySQL, SQLite, SQL, ChromaDB, Neo4j, Pinecone, AstraDB, and CassandraDB. In the AI/ML Tools category, you have experience with TensorFlow, Keras, Hugging Face, OpenAI, Gemini, Langchain, Langsmith, Ollama, Vertex AI, GROQ, Crew AI, and NVIDIA NIM. Your Tools and IDEs proficiency is equally impressive, with Git, GitHub, PowerBI, Streamlit, Streamlit Cloud, Bolt.new, Firebase Studio, FastAPI, VSCode, Jupyter, Google Colab, Anaconda, Replit, MySQL Workbench, PyCharm, and IntelliJ IDEA. and including soft skills i have worked for leadership,adaptability,Consistency,Team Collaboration,Punctualness in dedline completion and have worked on various projects such as built an impressive portfolio of projects showcasing expertise in AI, machine learning, data science, and full-stack development. These include Forest Fires Prediction, which applies ML to forecast wildfire risks using environmental data; AI Search Engine Agent, combining LangChain with RAG for intelligent, context-aware search; Churn Prediction App, helping businesses identify at-risk customers; and Insurance Prediction App, which estimates claims or premium costs for smarter underwriting. You created GenAI Resume Crafter to optimize resumes with AI, AI Chat with SQL Database for natural-language SQL querying, AI Chat with PDFs for conversational PDF interaction, AI Text Summarization for summarizing PDFs and YouTube transcripts, and AI Math Problem Solver for step-by-step solutions to math problems. You also developed Movie Review Sentiment Analysis for NLP-based opinion classification, NIM Chain RAG combining NVIDIA NIM and LangChain for high-performance search, Gemini AI Chatbot and Groq AI Chatbot for fast, context-aware responses, and language modeling tools like LSTM RNN Next Word Prediction and GRU RNN Next Word Prediction for intelligent text generation.highly driven and multi-skilled technology enthusiast with a proven track record of continuous learning and practical application across AI, cloud computing, data science, and software development. His credentials span prestigious organizations and platforms, including Google Cloud’s Gen AI Academy, IBM’s AI & Cloud internships, AWS Academy’s cloud programs, and LinkedIn–Microsoft’s Career Essentials in Generative AI. He has honed expertise in cutting-edge areas like Vertex AI, Gemini API, LLMs, data analytics, NLP, and enterprise-grade AI, complemented by strong foundations in Java-based Data Structures & Algorithms through Apna College’s Alpha program. I have also gained hands-on experience through internships and job simulations with Elevate Labs, Deloitte, and Forage in collaboration with Tata, tackling real-world projects in analytics, visualization, and business problem-solving. His certifications cover a broad spectrum—from Generative AI and ServiceNow enterprise platforms to comprehensive bootcamps in machine learning, deep learning, and NLP—demonstrating not only his technical depth but also his adaptability to emerging trends. With consistent 100% completion rates across 26-module programs and multiple projects per course,  exemplifies a commitment to excellence, innovation, and lifelong growth in the tech landscape.As a achievement i have secured 2nd rank in the codeathon competition oraganiized by IEEE at MGM also i have cleared the gate exam and receiverd a best performer tag in the elevate labs internship of Data analyst
"""


st.set_page_config(
    page_title="Rajat Ranvir Portfolio Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Rajat Ranvir Portfolio Chatbot")
st.caption("Ask me anything about Rajat's skills, projects, or experience")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I'm here to tell you about Rajat Ranvir's portfolio. What would you like to know?"}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Your question:"):

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    system_prompt = (
        f"You are an expert chatbot representing Rajat Ranvir's portfolio. "
        f"Answer the user's queries only using the information provided below. "
        f"Be concise but informative. If asked about something not mentioned, "
        f"respond with exactly: \"I don't have any idea about it\".\n\n"
        f"{rajat_info}"
    )


    messages_for_llm = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompt)
    ]

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        llm = ChatGroq(
            groq_api_key=api_key,
            model_name="llama3-70b-8192",
            streaming=True,
            temperature=0.3
        )

        for chunk in llm.stream(messages_for_llm):
            full_response += chunk.content
            message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})

with st.sidebar:
    st.header("About This Chatbot")
    st.markdown("""
    This chatbot knows about:
    - Rajat's technical skills
    - Education background
    - Projects portfolio
    - Work experience
    - Certifications and achievements
    """)
    st.divider()
    st.markdown("**Note:** Responses are limited to the provided knowledge base")