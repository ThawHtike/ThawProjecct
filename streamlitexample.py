
import os
import streamlit as st
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate

import langchain.llms
# Set up API key for DeepAI
api_key = "a5bedda26f30751cafc720db15d72c20"
os.environ["OPENAI_API_KEY"] = api_key

# Initialize the DeepAI model
llm = OpenAI(temperature=0.3)

# Title of the Streamlit app
st.title("Create Cover Letter")

# Display an image (ensure the path is correct)
st.image("thaw.png", width=100)

# Text area for user to input resume or CV
input_resume = st.text_area("Copy and paste your resume or CV here:")

# Text area for user to input the job description
input_job_description = st.text_area("Copy and paste the Job Description here:")

# Button to create the cover letter
if st.button("Create Cover Letter"):
    # Check if both inputs are provided
    if input_resume and input_job_description:
        # Prepare a prompt for the model
        prompt = f"Create a cover letter based on the following resume and job description:\n\nResume:\n{input_resume}\n\nJob Description:\n{input_job_description}"

        # Generate the cover letter
        cover_letter = llm(prompt)

        # Display the generated cover letter
        st.subheader("Generated Cover Letter")
        st.write(cover_letter)
    else:
        st.error("Please fill in both fields: Resume and Job Description.")

