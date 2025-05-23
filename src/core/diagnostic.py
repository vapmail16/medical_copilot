import os

def check_env_vars(required_vars):
    print("🔍 Checking Environment Variables...")
    for var in required_vars:
        if not os.getenv(var):
            print(f"❌ Missing: {var}")
        else:
            print(f"✅ Found: {var}")

def main():
    required_vars = [
        "OPENAI_API_KEY",
        "PERPLEXITY_API_KEY",
        "NEO4J_URI",
        "NEO4J_USER",
        "NEO4J_PASSWORD",
        "DEEPGRAM_API_KEY",
        "GOOGLE_APPLICATION_CREDENTIALS"  # For Google Cloud Vision and Speech APIs
    ]
    check_env_vars(required_vars)

if __name__ == "__main__":
    main()
