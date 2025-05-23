from typing import Dict, Any, List
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.render import format_tool_to_openai_function
from langchain_openai import ChatOpenAI

class SymptomExtractor:
    def __init__(self, model_name: str = "gpt-4"):
        self.llm = ChatOpenAI(model_name=model_name, temperature=0)
        self.tools = self._setup_tools()
        self.agent = self._setup_agent()

    def _setup_tools(self) -> List[Any]:
        """Setup tools for symptom extraction."""
        # TODO: Implement specific tools for symptom extraction
        return []

    def _setup_agent(self) -> AgentExecutor:
        """Setup the agent with tools and prompt."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a medical symptom extraction agent. Your task is to identify and extract relevant symptoms from patient input."),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        agent = (
            {
                "input": lambda x: x["input"],
                "agent_scratchpad": lambda x: format_to_openai_function_messages(x["intermediate_steps"]),
            }
            | prompt
            | self.llm
            | OpenAIFunctionsAgentOutputParser()
        )

        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True
        )

    async def extract_symptoms(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract symptoms from multi-modal input."""
        try:
            # Combine all input types into a single text
            combined_input = self._combine_inputs(input_data)
            
            # Run the agent
            result = await self.agent.ainvoke({"input": combined_input})
            
            return {
                "status": "success",
                "symptoms": result["output"],
                "confidence": result.get("confidence", 0.8)
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    def _combine_inputs(self, input_data: Dict[str, Any]) -> str:
        """Combine different input types into a single text."""
        combined = []
        
        for result in input_data.get("results", []):
            if result["status"] == "success":
                combined.append(f"{result['type']}: {result['content']}")
        
        return "\n".join(combined) 