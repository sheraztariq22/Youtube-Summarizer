"""
Test which Gemini model names work with your langchain-google-genai version
"""

import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("❌ Please set GEMINI_API_KEY in your .env file")
    exit(1)

os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

# Test different model names
model_names = [
    "gemini-pro",
    "models/gemini-pro",
    "gemini-1.0-pro",
    "models/gemini-1.0-pro",
    "gemini-1.5-flash",
    "models/gemini-1.5-flash",
]

print("Testing Gemini model names with LangChain...")
print("=" * 60)

from langchain_google_genai import ChatGoogleGenerativeAI

for model_name in model_names:
    print(f"\nTesting: {model_name}")
    print("-" * 60)
    
    try:
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=GEMINI_API_KEY,
            temperature=0.1
        )
        
        # Try a simple invoke
        response = llm.invoke("Say 'hello' in one word")
        print(f"✅ SUCCESS! Model works: {model_name}")
        print(f"   Response: {response.content}")
        
        # This is the one that works!
        print(f"\n🎯 USE THIS MODEL NAME: {model_name}")
        break
        
    except Exception as e:
        error_msg = str(e)
        if "404" in error_msg or "not found" in error_msg.lower():
            print(f"❌ Model not found: {model_name}")
        else:
            print(f"❌ Error: {e}")

print("\n" + "=" * 60)
print("Update your .env file with the working model name!")