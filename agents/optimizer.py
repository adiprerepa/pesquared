from langgraph.graph import END, StateGraph
from typing import List, Dict, TypedDict
from langchain_text_splitters import MarkdownHeaderTextSplitter
from llms.llm import UniversalLLM
from llms.prompts import (
    ALGORITHM_BOTTLENECK_PROMPT,
    ARCHITECTURE_BOTTLENECK_PROMPT,
    OPTIMIZATION_PROMPT
)

# Define the state type
class PerfState(TypedDict, total=False):
    function_name: str
    code: str
    bottlenecks: List[int]
    algorithm_steps: str
    algorithm_bottlenecks: str
    architecture_bottlenecks: str
    optimized_function: str
    algo_notes: Dict[int, str]
    arch_notes: Dict[int, str]

def get_top_bottleneck_lines(fn: str) -> List[int]:
    # TODO: replace with real analysis
    return [67, 58, 104]

def get_optimizer_agent(llm_model: str = "Llama-3.3-8B-Instruct", provider: str = "meta", temperature: float = 0):
    universal = UniversalLLM(model=llm_model, provider=provider, temperature=temperature)

    def analyze_algorithmic_bottlenecks(state: PerfState) -> PerfState:
        prompt = ALGORITHM_BOTTLENECK_PROMPT.format(
            function_name=state["function_name"],
            code=state["code"],
            bottleneck=state["code"]
        )
        result = universal.prompt(prompt)
        splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[
            ("#",    "Header 1"),
            ("##",   "Header 2"),
            ("###",  "Header 3"),
            ("####", "Notes"),
        ])
        splits = splitter.split_text(result)
        notes = [
            s.page_content.strip()
            for s in splits
            if s.metadata.get("Notes", "").lower().startswith("note to self")
        ]
        return {"algorithm_bottlenecks": "\n- ".join(notes).strip()}

    def analyze_architecture_bottlenecks(state: PerfState) -> PerfState:
        prompt = ARCHITECTURE_BOTTLENECK_PROMPT.format(
            function_name=state["function_name"],
            code=state["code"]
        )
        return {"architecture_bottlenecks": universal.prompt(prompt).strip()}

    def optimize_function(state: PerfState) -> PerfState:
        prompt = OPTIMIZATION_PROMPT.format(
            function_name=state["function_name"],
            code=state["code"],
            algorithm_bottlenecks=state.get("algorithm_bottlenecks",""),
            architecture_bottlenecks=state.get("architecture_bottlenecks",""),
        )
        result = universal.prompt(prompt)
        splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[
            ("#",    "Header 1"),
            ("##",   "Header 2"),
            ("###",  "Header 3"),
            ("####", "Notes"),
        ])
        splits = splitter.split_text(result)
        for s in splits:
            if state["function_name"].lower() in s.metadata.get("Header 1","").lower():
                return {**state, "optimized_function": s.page_content.strip()}
        return {**state, "optimized_function": result.strip()}

    def gen_bottlenecks(state: PerfState) -> PerfState:
        return {"bottlenecks": get_top_bottleneck_lines(state["function_name"])}

    def make_process_fn(idx: int):
        def process_one(state: PerfState) -> PerfState:
            ln = state["bottlenecks"][idx]
            # Placeholder: in real use, slice code around ln
            slice_code = state["code"]
            sub: PerfState = {
                "function_name": f"{state['function_name']} [line {ln}]",
                "code": slice_code
            }
            sub.update(analyze_algorithmic_bottlenecks(sub))
            if not sub.get("algorithm_bottlenecks"):
                sub.update(analyze_architecture_bottlenecks(sub))
            return {
                "algo_notes": {idx: sub.get("algorithm_bottlenecks","")},
                "arch_notes": {idx: sub.get("architecture_bottlenecks","")}
            }
        return process_one

    def aggregate_notes(state: PerfState) -> PerfState:
        algo = [
            f"**Line {ln}:**\n{state.get('algo_notes', {}).get(i, '')}"
            for i, ln in enumerate(state["bottlenecks"])
        ]
        arch = [
            f"**Line {ln}:**\n{state.get('arch_notes', {}).get(i, '')}"
            for i, ln in enumerate(state["bottlenecks"])
        ]
        return {
            "algorithm_bottlenecks": "\n\n".join(algo).strip(),
            "architecture_bottlenecks": "\n\n".join(arch).strip()
        }

    graph = StateGraph(PerfState)
    graph.set_entry_point("gen_bottlenecks")
    graph.add_node("gen_bottlenecks", gen_bottlenecks)

    def create_processors(state: PerfState) -> List[str]:
        num_bottlenecks = len(state["bottlenecks"])
        processor_names = []
        for i in range(num_bottlenecks):
            name = f"process_b{i}"
            graph.add_node(name, make_process_fn(i))
            graph.add_edge("gen_bottlenecks", name)
            processor_names.append(name)
        return processor_names

    graph.add_node("aggregate", aggregate_notes)

    def add_processor_edges(processor_names: List[str]):
        for name in processor_names:
            graph.add_edge(name, "aggregate")

    graph.add_node("optimize", optimize_function)
    graph.add_edge("aggregate", "optimize")
    graph.add_edge("optimize", END)

    dummy_state = {"bottlenecks": [0]}
    processor_names = create_processors(dummy_state)
    add_processor_edges(processor_names)

    return graph.compile()
