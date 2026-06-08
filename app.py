import os
import sys

# ==========================================
# PLATFORM-AGNOSTIC ENVIRONMENT OVERRIDE
# ==========================================
current_working_directory = os.path.dirname(os.path.abspath(__file__))

# ONLY inject local paths if running locally (not on Streamlit Cloud)
# Streamlit Cloud builds its path under '/home/adminuser/' or '/mount/src/'
if "adminuser" not in current_working_directory and "mount" not in current_working_directory:
    local_venv_packages = os.path.join(current_working_directory, ".venv", "Lib", "site-packages")
    if os.path.exists(local_venv_packages) and local_venv_packages not in sys.path:
        sys.path.insert(0, local_venv_packages)

# ==========================================
# STANDARD DEPENDENCY IMPORTS
# ==========================================
import asyncio
import json
from typing import List
import streamlit as st
from groq import Groq
from dotenv import load_dotenv
from playwright.async_api import async_playwright

# 1. Page Configuration & Setup
st.set_page_config(page_title="AI Agent Scraper", page_icon="🤖", layout="centered")
load_dotenv()

# Grab the Groq API Key purely from environment/secrets
api_key = os.environ.get("GROQ_API_KEY")

# Safety Interception: Only initialize if the key is present
if not api_key:
    st.error("🔒 GROQ_API_KEY environment variable is missing. Please configure it securely in your Streamlit Cloud Secrets dashboard or local .env file.")
    st.stop()  # Stops the app execution instantly so it doesn't crash downstream
else:
    client = Groq(api_key=api_key)

# 2. Define the Tool blueprint for Groq
tools_blueprint = [
    {
        "type": "function",
        "function": {
            "name": "display_extracted_job",
            "description": "Call this tool to format and return the extracted job listing components.",
            "parameters": {
                "type": "object",
                "properties": {
                    "job_title": {"type": "string", "description": "The formal title of the job role."},
                    "company_name": {"type": "string", "description": "The exact name of the hiring company."},
                    "salary": {"type": "string", "description": "The full compensation details or salary ranges given."},
                    "skills": {"type": "array", "items": {"type": "string"}, "description": "List of core technologies or skills required."},
                    "is_remote": {"type": "boolean", "description": "True if remote/WFH eligible, otherwise false."}
                },
                "required": ["job_title", "company_name", "salary", "skills", "is_remote"]
            }
        }
    }
]

# 3. Core AI Engine
def analyze_text_with_tools(raw_text: str) -> dict:
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system", 
                "content": "You are an automated extraction agent. Analyze the user's web text and call the 'display_extracted_job' tool with exact parameters."
            },
            {"role": "user", "content": f"Extract details from this text:\n\n{raw_text}"}
        ],
        tools=tools_blueprint,
        tool_choice={"type": "function", "function": {"name": "display_extracted_job"}},
        temperature=0.0
    )
    tool_call = response.choices[0].message.tool_calls[0]
    return json.loads(tool_call.function.arguments)

# 4. Asynchronous Scraper
async def scrape_live_url(url: str) -> str:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(url, wait_until="domcontentloaded", timeout=30000)
        raw_body_text = await page.locator("body").inner_text()
        await browser.close()
        return raw_body_text[:6000]  # Safe token limitation window

# 5. Streamlit Frontend Layout
st.title("🤖 Autonomous AI Agent Scraper")
st.subheader("Extract structured parameters from any text context instantly.")
st.write("This application leverages an asynchronous Playwright engine paired with a Llama-3 parsing framework via function calling tools.")

st.markdown("---")

# User input selection tabs
tab1, tab2 = st.tabs(["📝 Paste Raw Text / HTML", "🌐 Scrape Live URL"])

with tab1:
    user_text_input = st.text_area("Paste text or HTML layout content here:", height=250, placeholder="Example: Senior AI Developer wanted at Google. Compensation: $200k. Required skills: Python, PyTorch...")
    if st.button("Extract Data From Text", key="btn_text"):
        if user_text_input.strip() == "":
            st.warning("Please provide some text context first.")
        else:
            with st.spinner("Agent parsing text matrix via Groq LPUs..."):
                try:
                    extracted_data = analyze_text_with_tools(user_text_input)
                    
                    st.success("Extraction Complete!")
                    st.balloons()
                    
                    # Display metrics structural outputs
                    st.metric(label="💼 Job Title", value=extracted_data.get("job_title"))
                    st.metric(label="🏢 Company Name", value=extracted_data.get("company_name"))
                    st.metric(label="💰 Salary Package", value=extracted_data.get("salary"))
                    st.metric(label="🏠 Remote Workspace", value="Yes" if extracted_data.get("is_remote") else "No")
                    
                    st.write("🔑 **Extracted Tech Stack Tags:**")
                    st.write(", ".join([f"`{skill}`" for skill in extracted_data.get("skills", [])]))
                except Exception as e:
                    st.error(f"Execution boundary error: {str(e)}")

with tab2:
    user_url_input = st.text_input("Enter a live URL to scrape:", placeholder="https://wikipedia.org/wiki/Software_engineer")
    if st.button("Run Browser Agent", key="btn_url"):
        if user_url_input.strip() == "":
            st.warning("Please enter a valid target URL.")
        else:
            with st.spinner("Launching Chromium browser and grabbing data..."):
                try:
                    scraped_content = asyncio.run(scrape_live_url(user_url_input))
                    
                    with st.spinner("Analyzing text layout metrics..."):
                        extracted_data = analyze_text_with_tools(scraped_content)
                        
                        st.success("Dynamic Scrape & Parse Successful!")
                        
                        st.metric(label="💼 Job Title", value=extracted_data.get("job_title"))
                        st.metric(label="🏢 Company Name", value=extracted_data.get("company_name"))
                        st.metric(label="💰 Salary Package", value=extracted_data.get("salary"))
                        st.metric(label="🏠 Remote Workspace", value="Yes" if extracted_data.get("is_remote") else "No")
                        
                        st.write("🔑 **Extracted Tech Stack Tags:**")
                        st.write(", ".join([f"`{skill}`" for skill in extracted_data.get("skills", [])]))
                except Exception as e:
                    st.error(f"Browser action failed or timed out: {str(e)}")