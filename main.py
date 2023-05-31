# Import os to set API key
import os
import pyaudio
# # listen voice
import speech_recognition as speech
# # Python text to speech
import pyttsx3

import sys
import datetime
import json

# Import OpenAI as main LLM service
from langchain.llms import OpenAI
# Bring in streamlit for UI/app interface
import streamlit as st

# Import PDF document loaders...there's other ones as well!
from langchain.document_loaders import PyPDFLoader
# Import chroma as the vector store 
from langchain.vectorstores import Chroma


listener = speech.Recognizer()
machine = pyttsx3.init()

voices = machine.getProperty('voices')
# Change to female voice
machine.setProperty('voice', voices[0].id)
# Slow down the speed rate of voice
machine.setProperty('rate',180)

def talk(asked):
    machine.say(asked)
    machine.runAndWait()


# Import vector store stuff
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)

# Set APIkey for OpenAI Service
# Can sub this out for other LLM providers
os.environ['OPENAI_API_KEY'] = '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'

# Create instance of OpenAI LLM
llm = OpenAI(temperature=0.1, verbose=True)

# Create and load PDF Loader
loader = PyPDFLoader('New File For Analysis.pdf')
# Split pages from pdf 
pages = loader.load_and_split()
# Load documents into vector database aka ChromaDB
store = Chroma.from_documents(pages, collection_name='annualreport')
talk("Please wait, while I'm studying.")
# Create vectorstore info object - metadata repo?
vectorstore_info = VectorStoreInfo(
    name="annual_report",
    description="a banking annual report as a pdf",
    vectorstore=store
)
# Convert the document store into a langchain toolkit
toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)

# Add the toolkit to an end-to-end LC
agent_executor = create_vectorstore_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True
)
st.text("Alright, I've studied. How can I help you with this document.")
st.subheader('OpenAI 🦜🔗 Document Analyzer')
# Previous step : talk("Alright, I've studied. How can I help you with this document.")
talk("Alright, I've studied. Now, you can either type or say your question")

def ExecutionMethod():
    talk("You've 3 seconds to choose.")
    with speech.Microphone() as source:
        s_to_t = listener.listen(source)
        act_command = listener.recognize_google(s_to_t)
        prompt = act_command.lower()
        
        if 'type' in prompt:
            print("Asked: ", prompt)
            talk("Alright! Moving to type mode.")
            TypeMe()

        elif 'voice' in prompt:
            st.text('Moving with voice mode')
            AskMe()

        else:
            talk("Sorry, I couldn't catch it. Please say again.")
            ExecutionMethod()



def TypeMe():
    TypedPrompt = st.text_input('Input your prompt here')
    # If the user hits enter
    if TypedPrompt:
        Enumerate()
    
    else:
        TypeMe()


def AskMe():
    talk("Got you, opted voice mode. Please ask me your question.")
    with speech.Microphone() as source:
        s_to_t = listener.listen(source)
        try:
            act_command = listener.recognize_google(s_to_t, show_all=True)
            VoicePrompt = act_command.lower()
            # prompt = st.text_input(act_command)
            # print("prompt: ", prompt)
            st.text("You've asked : ", VoicePrompt)
            Enumerate()
        
        except Exception as e:
            st.text(e)
            # print(e)

def Enumerate():
    if TypedPrompt is not None:
        FinalPrompt = TypedPrompt

    elif VoicePrompt is not None:
        FinalPrompt = VoicePrompt
    
    else:
        talk("Sorry, let's try again")
        ExecutionMethod()

    # # Create a text input box for the user
    # with speech.Microphone() as source:
    #     s_to_t = listener.listen(source)
    #     act_command = listener.recognize_google(s_to_t)
    #     prompt = act_command.lower()
    #     # prompt = st.text_input(act_command)
    #     # print("prompt: ", prompt)
    #     print("Asked: ", prompt)
                    
        # prompt = st.text_input('Input your prompt here')
        
        # # If the user hits enter
        # if prompt:
        # Then pass the prompt to the LLM
    response = agent_executor.run(FinalPrompt)
    # ...and write it out to the screen
    st.write(response)
    

    # With a streamlit expander  
    with st.expander('Document Similarity Search'):
        # Find the relevant pages
        search = store.similarity_search_with_score(FinalPrompt) 
        # Write out the first 
        st.write(search[0][0].page_content) 
        talk(response)

    # else:
    #         pass
    

while True:
    ExecutionMethod()