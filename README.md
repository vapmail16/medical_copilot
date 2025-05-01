# 🏥 Medical Copilot

A multi-modal, multi-agent GenAI application for intelligent medical diagnosis, part of the broader **Knowledge Copilot** initiative.

## 🔧 Features
- Symptom intake via text, voice, or image
- Multi-agent orchestration with LangGraph
- Neo4j-powered contextual case reasoning
- Perplexity Sonar + LLM-as-a-Judge evaluation
- Toggle between Autonomous and Human-in-the-Loop modes
- Fully deployable on AWS

## 🚀 Quick Start
```bash
make install
make diagnose
make run

📁 Project Structure

medical_copilot/
│
├── .devcontainer/          # Dev container setup
├── diagnostic.py           # Environment checker
├── Makefile                # Developer automation
├── Dockerfile              # Docker environment
├── .env                    # Runtime secrets (excluded)
├── pyproject.toml          # Package config