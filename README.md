# Medical Copilot

A multi-agent AI health copilot using GenAI for medical diagnosis and assistance.

## Project Structure

```
medical_copilot/
├── src/                    # Source code
│   ├── core/              # Core functionality
│   ├── agents/            # LangGraph agents
│   ├── utils/             # Utility functions
│   └── api/               # API endpoints
├── tests/                 # Test files
├── config/                # Configuration files
├── data/                  # Data storage
└── docs/                  # Documentation
```

## Features

1. Multi-Modal Symptom Intake
   - Text input
   - Voice input (speech-to-text using Whisper or Deepgram)
   - Image input (using OpenAI Vision API)

2. Agentic Diagnosis Workflow
   - Symptom Extractor
   - Medical Knowledge Retriever
   - Risk Evaluator
   - Diagnosis Generator
   - Alternative Explanation Generator
   - LLM-as-a-Judge

3. Context-Aware Retrieval
   - Neo4j graph database integration
   - Similar case matching
   - Symptom relationship analysis

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/medical_copilot.git
cd medical_copilot
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -e .
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

5. Run the application:
```bash
python -m src.main
```

## Required Environment Variables

### Essential APIs
- `OPENAI_API_KEY`: OpenAI API key for text, vision, and chat models
  - Get it from: https://platform.openai.com/api-keys

### Optional APIs (Choose based on your needs)
1. Speech-to-Text (Choose one):
   - `DEEPGRAM_API_KEY`: For Deepgram speech-to-text
     - Get it from: https://deepgram.com/ (Free tier available)
   - Or use Whisper (included with OpenAI, no additional key needed)

2. Knowledge Validation:
   - `PERPLEXITY_API_KEY`: For fact-checking and validation
     - Get it from: https://www.perplexity.ai/ (Beta access required)

3. Graph Database:
   - `NEO4J_URI`: Neo4j database URI
   - `NEO4J_USER`: Neo4j username
   - `NEO4J_PASSWORD`: Neo4j password
   - Get these from: https://neo4j.com/ (Free tier available)

## Development

1. Install development dependencies:
```bash
pip install -e ".[dev]"
```

2. Run tests:
```bash
pytest
```

3. Run linting:
```bash
flake8
```

## License

MIT License

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request