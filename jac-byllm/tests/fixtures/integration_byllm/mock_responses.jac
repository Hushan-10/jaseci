"""Pre-configured MockLLM outputs for each integration test scenario.

Each load function populates the MockLLM output queues with the exact
sequence of responses needed for one test scenario. The outputs were
derived from real OpenAI API responses and translated into MockLLM format.
"""

import from byllm.lib { MockToolCall }
import from tools { calculate }


def load_concept_agent_outputs(llms: dict) {
    # Routing decision: visit ConceptAgent (index 0)
    llms["llm2"].config["outputs"].extend([
        [0],
    ]);
    # ConceptAgent.explain response
    llms["llm1"].config["outputs"].extend([
        "Machine learning is a way for computers to learn from data and improve their performance on tasks without being explicitly programmed. It involves using algorithms to find patterns in data, which can then be used to make predictions or decisions.",
    ]);
}


def load_math_agent_outputs(llms: dict) {
    # Routing decision: visit MathAgent (index 1)
    llms["llm2"].config["outputs"].extend([
        [1],
    ]);
    # MathAgent.solve uses ReAct with stream=True, logging=True and tools=[calculate]
    # Step 1: LLM decides to call the calculate tool
    # Step 2: LLM calls finish_tool with the result
    # Step 3: Final streamed answer after finish_tool
    llms["llm2"].config["outputs"].extend([
        MockToolCall(tool=calculate, args={"expr": "(15 + 5) * 2 - (10 / 2)"}),
        "35.0",
    ]);
}


def load_research_agent_outputs(llms: dict, ResearchSummary: type) {
    # Routing decision: visit ResearchAgent (index 2)
    llms["llm2"].config["outputs"].extend([
        [2],
    ]);
    # ResearchAgent.analyze returns a ResearchSummary object
    llms["llm2"].config["outputs"].extend([
        ResearchSummary(
            summary="This analysis compares supervised fine-tuning and prompt engineering as methods for adapting large language models to specific tasks. Supervised fine-tuning involves training a pre-existing model on a labeled dataset to improve its performance on that task, while prompt engineering focuses on crafting input prompts to elicit desired responses from the model without altering its weights.",
            key_points=[
                "Supervised fine-tuning typically requires a large labeled dataset and can significantly enhance model performance on specific tasks.",
                "Prompt engineering leverages the existing knowledge of the model and requires no additional training, making it quicker and more flexible.",
                "Fine-tuning may lead to overfitting if the dataset is not sufficiently diverse, while prompt engineering's effectiveness can vary based on prompt design.",
                "Supervised fine-tuning can be resource-intensive, whereas prompt engineering is often more accessible and cost-effective for rapid prototyping."
            ],
            confidence="High"
        ),
    ]);
}
