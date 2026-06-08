import os
from typing import List
from pydantic import BaseModel, Field
from groq import Groq  # Swapped from OpenAI to Groq for free practice tier
from dotenv import load_dotenv

load_dotenv()

# Grab the Groq API Key
api_key = os.environ.get("GROQ_API_KEY")

if not api_key:
    # Your exact Groq API Key safely handled as a single string
    api_key = os.environ.get("GROQ_API_KEY")

# Initialize the Groq client
client = Groq(api_key=api_key)

# Define the target structures using Pydantic V2
class ProductAttribute(BaseModel):
    key: str = Field(description="The structural property name, e.g., 'RAM', 'Storage', 'Battery'.")
    value: str = Field(description="The value of the property, e.g., '16GB', '1TB', '5000mAh'.")

class ExtractedProduct(BaseModel):
    product_name: str = Field(description="The formal brand name and model of the item.")
    price_usd: float = Field(description="The price converted to a floating-point USD value.")
    in_stock: bool = Field(description="True if explicitly available, False if out of stock or backordered.")
    specifications: List[ProductAttribute] = Field(description="List of key technical specifications found.")

# Simulated raw, messy data extracted from a web page
raw_scraped_html_text = """
<div class="product-container">
    <h1 class="title-3xl">TechSpecs UltraBook 14 Pro (Space Gray)</h1>
    <span class="price-tag">$1,299.99 Before taxes</span>
    <div class="inventory-status heavily-discounted">
        <span class="status-text">Hurry! Only 3 left in stock - order soon.</span>
    </div>
    <div class="specs-grid">
        <p>Equipped with the groundbreaking M4 Pro architecture featuring a 12-core CPU.</p>
        <p>Boasting a massive 32GB of unified memory unified for heavy LLM training workflows.</p>
        <p>Storage capacity maxes out at a blistering fast 1TB NVMe SSD.</p>
    </div>
</div>
"""

def extract_structured_web_data(raw_text: str) -> ExtractedProduct:
    # Tight prompt framing to force smaller open-source models to follow key structures
    system_prompt = (
        "You are an elite data extraction engine. You MUST output a single valid JSON object "
        "that matches the provided schema perfectly. Do not change key names.\n\n"
        "Your output JSON MUST contain exactly these top-level keys:\n"
        "- 'product_name' (string)\n"
        "- 'price_usd' (float number, extract numeric value only)\n"
        "- 'in_stock' (boolean true/false)\n"
        "- 'specifications' (array of objects containing 'key' and 'value')"
    )
    
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",  # Ultra-fast, highly capable open-source model
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Extract product details from this text into the requested schema:\n\n{raw_text}"}
        ],
        # Instructing Groq to parse output directly into our schema blueprint
        response_format={"type": "json_object", "schema": ExtractedProduct.model_json_schema()},
        temperature=0.0
    )
    
    # Extract the raw text JSON content string from Groq's response
    raw_json_string = response.choices[0].message.content
    
    # Hydrate it back into our typed Pydantic object
    return ExtractedProduct.model_validate_json(raw_json_string)

# Execute pipeline
try:
    print("[System] Initiating schema parsing pipeline via Groq Llama-3...")
    validated_data: ExtractedProduct = extract_structured_web_data(raw_scraped_html_text)
    
    print("\n--- Success! Data Extracted & Validated ---")
    print(f"Product Name:    {validated_data.product_name}")
    print(f"Price USD:       ${validated_data.price_usd}")
    print(f"In Stock Status: {validated_data.in_stock}")
    print("\nSpecifications Matrix:")
    for spec in validated_data.specifications:
        print(f" - {spec.key}: {spec.value}")
        
except Exception as e:
    print(f"\n[Error] Extraction Failed: {str(e)}")