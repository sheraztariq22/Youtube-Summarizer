"""
List all available Gemini models for your API key
"""

import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("❌ Please set GEMINI_API_KEY in your .env file")
    exit(1)

genai.configure(api_key=GEMINI_API_KEY)

print("=" * 70)
print("Available Gemini Models for Your API Key")
print("=" * 70)

try:
    models = genai.list_models()
    
    print("\n✅ Available models:\n")
    
    for model in models:
        print(f"📌 {model.name}")
        print(f"   Display name: {model.display_name}")
        print(f"   Description: {model.description[:80]}...")
        
        # Check if it supports generateContent
        if 'generateContent' in model.supported_generation_methods:
            print(f"   ✅ Supports text generation")
            print(f"   🎯 USE THIS: {model.name}")
        
        print()
    
    print("=" * 70)
    print("\nTo use in your app, update .env with one of the model names above")
    print("Example: GEMINI_MODEL=models/gemini-1.5-flash")
    
except Exception as e:
    print(f"\n❌ Error listing models: {e}")
    print("\nThis might mean:")
    print("1. Your API key is invalid")
    print("2. Your API key doesn't have access to Gemini models")
    print("3. Network/firewall issue")
    print("\nTry visiting: https://makersuite.google.com/app/apikey")
    print("And regenerate your API key")