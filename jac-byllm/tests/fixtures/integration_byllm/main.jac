"""Main entry point for integration."""
import os;
import from byllm.lib { Model, Image, Video }
import from tools{calculate}

glob llms = {
    "llm1": Model(
        model_name="gpt-4o-mini",config={"verbose": False},
    ),

    "llm2": Model(
        model_name="gpt-4o-mini",config={"verbose": False},
    ),

    "llm3": Model(
        model_name="gpt-4o-mini",config={"verbose": False}
    ),
};

# enums

enum Correctness { CORRECT, PARTIAL, WRONG }

enum Intent {
    ConceptAgent,
    MathAgent,
    ImageAgent,
    VideoAgent,
    ResearchAgent
}

enum ConfidenceLevel { LOW, MEDIUM, HIGH }

enum Personality {
    INTROVERT,
    EXTROVERT,
    AMBIVERT
}


obj StudyAnswer {
    has content: str,
        correctness: Correctness,
        confidence: float;
}

obj ResearchSummary {
    has summary: str,
        key_points: list[str],
        confidence: str;
}

obj Person {
    has full_name: str,
        year_of_death: str,
        personality: str;
}

node ConceptAgent {

    def explain(topic: str) -> str by llms["llm1"](
        stream=True
    );

    can handle with Supervisor entry {
        resp = self.explain(visitor.query);

        # Store result on the walker for it to report later
        visitor.agent_result = {"agent": "ConceptAgent", "response": resp};
    }
}

node MathAgent {

    def solve(problem: str) -> str by llms["llm2"](
        method="ReAct",
        tools=[calculate],
        max_react_iterations=3
    );

    can handle with Supervisor entry {
        result = self.solve(visitor.query);

        # Store result on the walker for it to report later
        visitor.agent_result = {"agent": "MathAgent", "response": result};
    }
}

node ResearchAgent {

    def analyze(query: str) -> ResearchSummary by llms["llm2"](
        method="Reason"
    );

    can handle with Supervisor entry {
        result = self.analyze(visitor.query);

        # Store result on the walker for it to report later
        visitor.agent_result = {
            "agent": "ResearchAgent",
            "summary": result.summary,
            "key_points": result.key_points,
            "confidence": result.confidence
        };
    }
}

walker:pub Supervisor {

    has query: str;

    can route with Root entry {

        if not [root --> (?:ConceptAgent)] { root ++> ConceptAgent(); }
        if not [root --> (?:MathAgent)] { root ++> MathAgent(); }
        if not [root --> (?:ResearchAgent)] { root ++> ResearchAgent(); }

        # Initialize result storage (dynamic attribute, not exposed in API)
        self.agent_result = None;

        visit [-->] by llms["llm2"](
            show_params=True,
            verbose=False,
            incl_info={
                "instruction": "Select exactly ONE agent node to visit. Return only ONE node.",
                "user_query": self.query,
            }
        );
    }

    can collect_result with Root exit {
        # Report collected results - this works in REST API
        if self.agent_result is not None {
            report self.agent_result;
            del self.agent_result;  # Remove from walker to avoid duplication in response
        }
    }
}
