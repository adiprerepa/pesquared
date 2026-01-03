"""Builder agent that architects C++ solutions using a multi-step graph-based approach."""

from langgraph.graph import END, StateGraph
from typing import TypedDict, List, Dict
from langchain.prompts import PromptTemplate
from langchain_text_splitters import MarkdownHeaderTextSplitter
from llms.llm import UniversalLLM

# Analysis prompt to understand requirements and constraints
ANALYSIS_PROMPT = PromptTemplate(
    input_variables=["problem_description"],
    template="""You are an expert C++ software architect. Analyze the following problem and identify key requirements and constraints:

Problem Description:
{problem_description}

Provide your analysis in the following markdown format:

## Requirements
- List key functional and non-functional requirements
- Include any specific performance targets
- Note any constraints or limitations

## Performance Constraints
- Specify target time complexity
- Specify target space complexity
- List any specific performance requirements

## Challenges
- List main technical challenges
- Identify potential implementation difficulties
- Note any architectural concerns

## Edge Cases
- List important edge cases to handle
- Note any special input conditions
- Include boundary conditions

## Bottlenecks
- Identify potential performance bottlenecks
- List resource constraints
- Note any scalability concerns"""
)

# Design prompt to propose data structures and patterns
DESIGN_PROMPT = PromptTemplate(
    input_variables=["problem_description", "analysis"],
    template="""Based on the problem description and analysis, propose the optimal data structures and design patterns:

Problem Description:
{problem_description}

Analysis:
{analysis}

Provide your design in the following markdown format:

## Data Structures
- List and explain each proposed data structure
- Include the purpose and benefits of each
- Note any trade-offs considered

## Class Hierarchy
- Describe the class structure and relationships
- Explain inheritance and composition choices
- List key methods and their purposes

## Design Patterns
- List applicable design patterns
- Explain why each pattern was chosen
- Describe how they solve specific problems

## Interfaces
- Define key interfaces
- Explain their responsibilities
- Note any design constraints"""
)

# Implementation prompt to generate the actual code
IMPLEMENTATION_PROMPT = PromptTemplate(
    input_variables=["problem_description", "analysis", "design"],
    template="""Based on the problem description, analysis, and design decisions, implement the complete C++ solution:

Problem Description:
{problem_description}

Analysis:
{analysis}

Design:
{design}

Provide your implementation in the following markdown format:

## Class Definitions
```cpp
// Include all necessary class definitions here
// With detailed comments explaining the design
```

## Core Algorithm
```cpp
// Implement the main algorithm
// Include detailed comments explaining the logic
```

## Helper Functions
```cpp
// Include all helper functions
// With clear documentation
```

## Error Handling
```cpp
// Include error handling code
// With appropriate error types and messages
```

## Usage Example
```cpp
// Provide a complete usage example
// Show how to use the implemented solution
```"""
)

class BuildState(TypedDict, total=False):
    problem_description: str
    analysis: Dict[str, str]
    design: Dict[str, str]
    implementation: str
    current_step: str

def get_builder_agent(llm_model: str = "Llama-3.3-8B-Instruct", provider: str = "meta", temperature: float = 0):
    universal = UniversalLLM(model=llm_model, provider=provider, temperature=temperature)
    
    # Initialize the markdown splitter
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[
        ("##", "Section"),
        ("###", "Subsection"),
    ])

    def analyze_problem(state: BuildState) -> BuildState:
        prompt = ANALYSIS_PROMPT.format(
            problem_description=state["problem_description"]
        )
        result = universal.prompt(prompt)
        # Split the markdown into sections
        splits = splitter.split_text(result)
        # Convert splits into a dictionary
        sections = {s.metadata["Section"]: s.page_content.strip() 
                   for s in splits if "Section" in s.metadata}
        return {"analysis": sections, "current_step": "analysis"}

    def design_solution(state: BuildState) -> BuildState:
        prompt = DESIGN_PROMPT.format(
            problem_description=state["problem_description"],
            analysis=state["analysis"]
        )
        result = universal.prompt(prompt)
        # Split the markdown into sections
        splits = splitter.split_text(result)
        # Convert splits into a dictionary
        sections = {s.metadata["Section"]: s.page_content.strip() 
                   for s in splits if "Section" in s.metadata}
        return {"design": sections, "current_step": "design"}

    def implement_solution(state: BuildState) -> BuildState:
        prompt = IMPLEMENTATION_PROMPT.format(
            problem_description=state["problem_description"],
            analysis=state["analysis"],
            design=state["design"]
        )
        result = universal.prompt(prompt)
        return {"implementation": result.strip(), "current_step": "implementation"}

    # Create the graph
    graph = StateGraph(BuildState)
    
    # Add nodes for each step
    graph.add_node("analyze", analyze_problem)
    graph.add_node("design", design_solution)
    graph.add_node("implement", implement_solution)
    
    # Set the entry point
    graph.set_entry_point("analyze")
    
    # Define the flow
    graph.add_edge("analyze", "design")
    graph.add_edge("design", "implement")
    graph.add_edge("implement", END)

    return graph.compile()
