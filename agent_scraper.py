import os
import asyncio
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator
from groq import Groq
from dotenv import load_dotenv
from playwright.async_api import async_playwright

# 1. Fire up dotenv to read environmental parameters
load_dotenv()

# 2. Extract and configure the API credentials
api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    # Fallback to your working verified practice key string
    api_key = os.environ.get("GROQ_API_KEY")

# 3. Instantiate the Groq client
client = Groq(api_key=api_key)

# 4. Define an elite, crash-resilient Pydantic validation schema
class JobExtractionSchema(BaseModel):
    job_title: Optional[str] = Field(default="Not Specified", description="The formal title of the job role, or null if not found.")
    company_name: Optional[str] = Field(default="Unknown Company", description="The name of the hiring organization, or null if not found.")
    estimated_salary: Optional[str] = Field(default="Not Specified", description="The posted compensation package, or null if not found.")
    required_skills: List[str] = Field(default_factory=list, description="A clean list of tech stack requirements or keywords found. Return an empty array [] if none found.")
    is_remote: Optional[bool] = Field(default=False, description="True if the position explicitly specifies remote/WFH options, otherwise False.")

    # Emergency field validators to intercept and correct 'null' returns from smaller open-source models
    @field_validator('required_skills', mode='before')
    @classmethod
    def ensure_list(cls, v):
        if v is None:
            return []
        return v

    @field_validator('is_remote', mode='before')
    @classmethod
    def ensure_bool(cls, v):
        if v is None:
            return False
        return v

# 5. Core inference parsing function
def process_page_content(raw_text: str) -> JobExtractionSchema:
    print("[Agent] Passing raw text layers to Llama-3 parsing engine...")
    
    system_prompt = (
        "You are an elite automated scraping engine. Extract job listing data from the text into valid JSON.\n"
        "CRITICAL RULES:\n"
        "1. Match the requested JSON schema keys perfectly.\n"
        "2. If 'required_skills' is missing, set it to an empty array [].\n"
        "3. If 'is_remote' is unknown, set it to false.\n"
        "4. For any other missing text field, set its value to null."
    )
    
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Analyze this web page content and isolate job details:\n\n{raw_text}"}
        ],
        # Hooking Groq directly into the structural JSON schema layout
        response_format={"type": "json_object", "schema": JobExtractionSchema.model_json_schema()},
        temperature=0.0
    )
    
    raw_json = response.choices[0].message.content
    return JobExtractionSchema.model_validate_json(raw_json)

# 6. Main execution engine managing asynchronous browser orchestration
async def run_automation_agent():
    async with async_playwright() as p:
        print("[Agent] Launching background Chromium engine...")
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        print("[Agent] Overriding viewport with a live mock job-board context...")
        mock_job_html = """
        <html>
            <body>
                <div class="job-header">
                    <h1>Senior AI Software Engineer (Remote Eligible)</h1>
                    <h2>Meta Platforms, Inc.</h2>
                </div>
                <div class="job-body">
                    <p>We are seeking a 3rd year or graduate software expert to scale automated agent workflows.</p>
                    <p>Compensation target: $145,000 - $185,000 base salary per year.</p>
                    <div class="tags">
                        <span>Required Stack: Python, JavaScript, Node.js, Pydantic, Playwright, Docker, LLM Fine-Tuning</span>
                    </div>
                </div>
            </body>
        </html>
        """
        
        # Inject the structured HTML mock directly into the browser DOM landscape
        await page.set_content(mock_job_html)
        
        print("[Agent] Scraping inner text content matrix...")
        raw_body_text = await page.locator("body").inner_text()
        
        # Explicitly shutdown browser worker instances cleanly
        await browser.close()
        
        # Fire text data into our structured parsing framework
        extracted_data = process_page_content(raw_body_text)
        
        # 7. Print the extracted, validated data parameters
        print("\n--- Production Agent Extraction Success ---")
        print(f"Role Title:      {extracted_data.job_title}")
        print(f"Company:         {extracted_data.company_name}")
        print(f"Salary Package:  {extracted_data.estimated_salary}")
        print(f"Core Tech Stack: {extracted_data.required_skills}")
        print(f"Remote Work:     {extracted_data.is_remote}")

# 8. Script runner wrapper
if __name__ == "__main__":
    asyncio.run(run_automation_agent())