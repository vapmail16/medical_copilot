import os
from dotenv import load_dotenv
import openai
from openai import OpenAI
from deepgram import Deepgram
import requests
from neo4j import GraphDatabase
import asyncio

def test_openai():
    """Test OpenAI API (including Whisper and Vision)"""
    print("\n🔍 Testing OpenAI API...")
    try:
        client = OpenAI()
        # Test basic chat completion
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello, this is a test."}],
            max_tokens=5
        )
        print("✅ OpenAI Chat API: Working")
        
        # Test if we can access vision model
        models = client.models.list()
        vision_model = any(model.id == "gpt-4-vision-preview" for model in models.data)
        if vision_model:
            print("✅ OpenAI Vision API: Available")
        else:
            print("❌ OpenAI Vision API: Not available")
            
        return True
    except Exception as e:
        print(f"❌ OpenAI API Error: {str(e)}")
        return False

async def test_deepgram():
    """Test Deepgram API"""
    print("\n🔍 Testing Deepgram API...")
    try:
        deepgram = Deepgram(os.getenv("DEEPGRAM_API_KEY"))
        # Test API key by making a simple transcription request
        with open("test.wav", "wb") as f:
            f.write(b"test")  # Create a dummy audio file
        
        with open("test.wav", "rb") as audio:
            source = {"buffer": audio, "mimetype": "audio/wav"}
            response = await deepgram.transcription.prerecorded(source, {
                "smart_format": True,
                "model": "nova",
            })
            print("✅ Deepgram API: Working")
            return True
    except Exception as e:
        print(f"❌ Deepgram API Error: {str(e)}")
        print("\nTroubleshooting steps:")
        print("1. Verify your Deepgram API key in .env file")
        print("2. Make sure you're using the correct API key format")
        return False
    finally:
        if os.path.exists("test.wav"):
            os.remove("test.wav")

def test_perplexity():
    """Test Perplexity API"""
    print("\n🔍 Testing Perplexity API...")
    try:
        from openai import OpenAI
        api_key = os.getenv('PERPLEXITY_API_KEY')
        if not api_key:
            print("❌ Perplexity API Error: API key not found in environment variables")
            return False

        client = OpenAI(api_key=api_key, base_url="https://api.perplexity.ai")
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an artificial intelligence assistant and you need to "
                    "engage in a helpful, detailed, polite conversation with a user."
                ),
            },
            {
                "role": "user",
                "content": "How many stars are in the universe?",
            },
        ]
        response = client.chat.completions.create(
            model="sonar-pro",
            messages=messages,
            max_tokens=10,
        )
        print("✅ Perplexity API: Working")
        return True
    except Exception as e:
        print(f"❌ Perplexity API Error: {str(e)}")
        return False

def test_neo4j():
    """Test Neo4j Connection"""
    print("\n🔍 Testing Neo4j Connection...")
    try:
        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USER")
        password = os.getenv("NEO4J_PASSWORD")
        
        print(f"Attempting to connect to: {uri}")
        print(f"Using user: {user}")
        
        driver = GraphDatabase.driver(
            uri,
            auth=(user, password)
        )
        # Test connection
        with driver.session() as session:
            result = session.run("RETURN 1 as test")
            record = result.single()
            if record and record["test"] == 1:
                print("✅ Neo4j Connection: Working")
                return True
    except Exception as e:
        print(f"❌ Neo4j Connection Error: {str(e)}")
        print("\nTroubleshooting steps:")
        print("1. Make sure Neo4j Desktop is installed and running")
        print("2. Check if the database is started")
        print("3. Verify your connection details in .env file:")
        print(f"   NEO4J_URI={uri}")
        print(f"   NEO4J_USER={user}")
        print("   NEO4J_PASSWORD=******")
        return False
    finally:
        if 'driver' in locals():
            driver.close()

async def main():
    # Load environment variables
    load_dotenv()
    
    print("🚀 Starting API Tests...")
    
    # Test each API
    openai_ok = test_openai()
    deepgram_ok = await test_deepgram()
    perplexity_ok = test_perplexity()
    neo4j_ok = test_neo4j()
    
    # Print summary
    print("\n📊 Test Summary:")
    print(f"OpenAI API: {'✅' if openai_ok else '❌'}")
    print(f"Deepgram API: {'✅' if deepgram_ok else '❌'}")
    print(f"Perplexity API: {'✅' if perplexity_ok else '❌'}")
    print(f"Neo4j Connection: {'✅' if neo4j_ok else '❌'}")
    
    # Overall status
    all_ok = all([openai_ok, deepgram_ok, perplexity_ok, neo4j_ok])
    print(f"\n{'🎉 All APIs are working correctly!' if all_ok else '⚠️ Some APIs failed. Please check the errors above.'}")

if __name__ == "__main__":
    asyncio.run(main()) 