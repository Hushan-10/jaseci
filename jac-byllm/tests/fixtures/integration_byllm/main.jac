"""Main entry point for integration."""

# Client-side imports (useState is auto-injected when using `has` variables)
cl import from react { useEffect }
cl import from .components.Button { Button }

import os;
import from byllm.lib { Model, Image, Video }
import from tools { calculate, run_and_test_python_code }

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
    CodeAgent,
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
        stream=False
    );

    can handle with Supervisor entry {
        print("📘 ConceptAgent selected\n");

        resp = self.explain(visitor.query);
        print(resp);

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
        print("🧮 MathAgent selected\n");
        result = self.solve(visitor.query);
        print("Math Result:", result);

        # Store result on the walker for it to report later
        visitor.agent_result = {"agent": "MathAgent", "response": result};
    }
}

node CodeAgent {

    def generate_code(task: str) -> str by llms["llm2"](
        method="ReAct",
        tools=[run_and_test_python_code],
        max_react_iterations=3
    );

    can handle with Supervisor entry {
        print("💻 CodeAgent selected\n");
        code = self.generate_code(visitor.query);
        print("Verified Code:\n");
        print(code);

        visitor.agent_result = {"agent": "CodeAgent", "response": code};
    }
}


node ResearchAgent {

    def analyze(query: str) -> ResearchSummary by llms["llm2"](
        method="Reason"
    );

    can handle with Supervisor entry {
        print("🧪 ResearchAgent selected\n");

        result = self.analyze(visitor.query);

        print("Research Summary:\n");
        print(result.summary);

        print("\nKey Points:");
        for p in result.key_points {
            print("- " + p);
        }

        print("\nConfidence Level:", result.confidence);

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
        if not [root --> (?:CodeAgent)] { root ++> CodeAgent(); }
        if not [root --> (?:ResearchAgent)] { root ++> ResearchAgent(); }

        # Initialize result storage (dynamic attribute, not exposed in API)
        self.agent_result = None;

        visit [-->] by llms["reason"](
            show_params=True,
            verbose=True,
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
