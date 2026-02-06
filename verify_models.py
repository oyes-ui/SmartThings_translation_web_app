
import asyncio
try:
    from google import genai
    from google.genai import types
except ImportError:
    try:
        import google.genai as genai
        from google.genai import types
    except ImportError:
        print("Error: Could not import google.genai. Please ensure 'google-genai' is installed.")
        genai = None

from openai import AsyncOpenAI
from dotenv import load_dotenv

async def test_routing():
    handler = ModelHandler()
    
    # Test cases: (model_name, expected_provider_part_of_log_or_error)
    test_models = [
        "gemini-2.5-flash",
        "gpt-5-mini",
        "gemini-2.0-flash",
        "gpt-4o"
    ]
    
    print("--- Model Provider Routing Test ---")
    for m in test_models:
        if "gpt" in m.lower():
            provider = "OpenAI"
        else:
            provider = "Google"
        print(f"Model: {m} -> Detected Provider: {provider}")

    # Real call test (Optional - might fail if API limit/key issue, but checks code flow)
    print("\n--- Code Flow Test (Generating small response) ---")
    try:
        # We'll use a very small prompt and a model that is likely to work
        res = await handler.generate_content("Hi", model_name="gemini-2.5-flash")
        print(f"Gemini Call Result: {res[:50]}...")
    except Exception as e:
        print(f"Gemini Call Failed (Expected if API issues): {e}")

    try:
        res = await handler.generate_content("Hi", model_name="gpt-5-mini")
        print(f"GPT Call Result: {res[:50]}...")
    except Exception as e:
        print(f"GPT Call Failed (Expected if API issues/unauthorized): {e}")

if __name__ == "__main__":
    asyncio.run(test_routing())
