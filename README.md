# PESquared - Performance Engineering with LLMs

PESquared is a research and demonstration platform for automatic performance engineering using Large Language Models (LLMs). It combines profiling data with LLM-driven code analysis to intelligently discover and apply performance optimizations.

## 🎯 Project Overview

PESquared provides two complementary optimization approaches:

1. **C++ Optimization Pipeline** (Original): Analyzes stack traces, optimizes entire call chains in C++ codebases, and uses Git-based workflow for managing optimization branches.

2. **Python Optimization Pipeline** (Research/Demo): A lightweight, self-contained demonstration of LLM-guided performance engineering for Python functions. Perfect for understanding the core concepts and experimenting with the approach.

## 🚀 Features

### C++ Pipeline (Production-oriented)
- **Call Chain Optimization**: Analyzes entire call chains to optimize both leaf functions and their parents
- **Iterative Performance Improvement**: Applies optimizations gradually, guided by measured performance gains
- **Intelligent Optimization Selection**: Uses LLMs to generate optimizations tailored to your specific codebase
- **Automatic Error Correction**: Detects compilation errors and prompts the LLM to fix them automatically
- **Performance Verification**: Validates that optimizations actually improve performance before accepting them
- **Git Integration**: Creates branches for each optimization, making it easy to review and manage changes

### Python Pipeline (Research/Demo-oriented)
- **Simple Profiling**: Profile Python functions using cProfile to identify bottlenecks
- **LLM-Guided Optimization**: Generate optimization suggestions using OpenAI or dummy client (no API key needed)
- **Correctness Verification**: Automatically verify optimized code produces correct results
- **Performance Measurement**: Compare before/after performance with statistical analysis
- **Interactive Notebooks**: Jupyter notebooks demonstrating the full workflow
- **CLI Demo Scripts**: Command-line tools for quick experimentation

## 🔧 How It Works

PESquared takes a systematic approach to performance optimization by combining profiling data with LLM-driven code improvements.

### Performance Hotspot Identification

The system begins by analyzing folded stack traces through the `StackAnalyzer` component. These traces capture the runtime behavior of your application during representative workloads. The analyzer identifies the most time-consuming functions (hotspots) and, crucially, their call chains - the sequence of function calls that lead to these hotspots. This contextual information is vital for making intelligent optimization decisions.

### Dependency Analysis and Context Extraction

Before optimization can begin, PESquared needs to understand both the function to be optimized and its surrounding context. The `DependencyExtractor` component:

- Locates the function's implementation in your codebase
- Identifies all types and data structures used by the function
- Maps out the function's call graph (what it calls and what calls it)
- Extracts location information for accurate code replacement

This comprehensive context allows the LLM to understand how the function operates within the larger system.

### LLM-Driven Optimization

With the function and its context in hand, PESquared uses its `FunctionOptimizer` component to craft a detailed prompt for the LLM (either OpenAI's GPT models or Anthropic's Claude). The prompt includes:

- The original function implementation
- All dependent type definitions
- Call hierarchy information (parent/child relationships)
- Specific optimization guidance based on the function's role in the call chain

For parent functions that call into hotspots, the system specifically instructs the LLM to reduce the frequency of calls to the expensive child functions.

### Iterative Improvement with Performance Verification

Rather than blindly accepting LLM suggestions, PESquared creates a git branch for each optimization attempt and scientifically verifies its impact:

1. The system applies the suggested optimization to the codebase
2. A custom `PerformanceVerifier` implementation measures the performance on both the original and optimized code
3. Only if the optimization shows measurable improvement does it become the new baseline

If compilation errors occur during this process, PESquared automatically feeds the error messages back to the LLM along with the original function, enabling it to correct its approach.

### Call Chain Propagation

What sets PESquared apart is its holistic approach to optimization. Instead of treating functions in isolation, it understands their relationships in call chains. The system optimizes entire chains starting from root functions, which often yields greater performance improvements than just optimizing leaf functions. This approach can eliminate unnecessary function calls altogether rather than merely making individual functions faster.

---

## 🐍 Python Optimization Pipeline (Research/Demo)

The Python pipeline provides a simpler, self-contained demonstration of LLM-guided performance engineering. It's designed for research, education, and experimentation.

### Quick Start (Python Pipeline)

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Run demos without any API keys (offline mode):**
```bash
# Run baseline profiling demo
python scripts/run_demo.py --demo baseline

# Run LLM optimization demo (uses dummy client)
python scripts/run_demo.py --demo llm
```

3. **Optional: Use real LLM (requires OpenAI API key):**
```bash
export OPENAI_API_KEY="your-api-key-here"
python scripts/run_demo.py --demo llm --use-real-llm --model gpt-3.5-turbo
```

4. **Or use Jupyter notebooks:**
```bash
jupyter notebook notebooks/01_baseline_profiling_demo.ipynb
jupyter notebook notebooks/02_llm_guided_optimization.ipynb
```

### Python Pipeline Architecture

The Python pipeline consists of four main modules:

1. **`analyzers/profiler.py`**: Profile Python functions using cProfile
   - `profile_function()`: Profile a callable and return structured results
   - `ProfileSummary`: Dataclass containing profiling statistics
   - `format_profile_summary()`: Format results for humans or LLMs

2. **`optimizers/llm_client.py`**: Abstract LLM client interface
   - `LLMClient`: Abstract base class
   - `DummyLLMClient`: Offline client for testing (no API key needed)
   - `OpenAILLMClient`: Real OpenAI integration (requires API key)

3. **`optimizers/optimizer.py`**: Orchestrate the optimization pipeline
   - `optimize_with_llm()`: Main function to profile → prompt → optimize
   - `construct_optimization_prompt()`: Build LLM-ready prompts
   - `OptimizationResult`: Dataclass with optimization results

4. **`verifiers/verifier.py`**: Verify correctness and measure performance
   - `measure_runtime()`: Measure execution time with statistics
   - `check_correctness()`: Verify optimized code produces correct results
   - `compare_performance()`: Compare original vs optimized performance

### Example: Basic Usage

```python
from optimizers.optimizer import optimize_with_llm
from optimizers.llm_client import create_llm_client
from verifiers.verifier import check_correctness, compare_performance

# Define your function
def slow_function(n):
    result = []
    for i in range(n):
        if i % 2 == 0:
            result.append(i ** 2)
    return result

# Create LLM client (dummy for testing, or 'openai' for real)
client = create_llm_client('dummy')

# Optimize
result = optimize_with_llm(slow_function, 10000, llm_client=client)

print("Explanation:", result.explanation)
print("Optimized code:", result.optimized_code)

# Test it (with appropriate safety measures in production)
# exec(result.optimized_code, namespace)
# optimized_func = namespace['slow_function']
# check correctness and performance...
```

### Demos and Notebooks

**Notebooks** (in `notebooks/`):
- `01_baseline_profiling_demo.ipynb`: Learn profiling basics with intentionally slow functions
- `02_llm_guided_optimization.ipynb`: Complete end-to-end LLM optimization workflow

**CLI Demos** (via `scripts/run_demo.py`):
- `--demo baseline`: Show profiling and manual optimization
- `--demo llm`: Run full LLM-guided optimization pipeline
- `--demo prompt`: Generate an LLM-ready prompt for a custom function

### Limitations and Safety

⚠️ **Important considerations for the Python pipeline:**

1. **Research/Demo Purpose**: This is designed for learning and experimentation, not production use
2. **Code Execution Safety**: Using `exec()` on LLM-generated code can be dangerous
   - Only use with trusted LLMs and in isolated environments
   - Review generated code before execution
   - Consider sandboxing/containerization for production
3. **Function Scope**: Best suited for small, self-contained functions
4. **LLM Reliability**: LLMs may:
   - Change function behavior (breaking correctness)
   - Generate code that doesn't run
   - Miss optimization opportunities
   - Suggest micro-optimizations with no real impact
5. **No Sandboxing**: The dummy client is safe, but real LLM outputs need human review

### Future Directions (Python Pipeline)

- Sandboxed execution environment for generated code
- Support for more LLM providers (Anthropic, local models)
- Automated test generation
- Batch optimization of multiple functions
- Integration with existing test frameworks
- Performance regression detection
- More sophisticated prompt engineering

---

## 📋 Requirements (C++ Pipeline)

```
clang
tiktoken
pandas>=2.0.0
lxml>=4.9.0
openai>=1.0.0
anthropic>=0.46.0
gitpython>=3.1.0
python-dotenv>=1.0.0
tabulate
```

## 🔍 Usage (C++ Pipeline)

Basic usage requires a directory containing your codebase and a directory containing folded stack files:

```bash
python main.py [codebase_dir] [stacks_dir] [options]
```

### Options

- `--num-functions`: Number of top functions to optimize (default: 3)
- `--correction-attempts`: Maximum attempts to correct compilation errors (default: 2)
- `--debug`: Enable debug output
- `--model`: LLM model to use (default: gpt-3.5-turbo)
- `--provider`: API provider to use ('openai' or 'anthropic')
- `--temperature`: Temperature for sampling from the model (default: 0.7)

## 🔐 Environment Setup (C++ Pipeline)

Create a `.env` file with your API keys:

```
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
```

## 🛠️ Architecture (C++ Pipeline)

PESquared C++ pipeline consists of several interconnected components:

- **StackAnalyzer**: Processes stack trace data to identify performance hotspots
- **DependencyExtractor**: Extracts function dependencies and context
- **FunctionOptimizer**: Generates optimized implementations using LLMs
- **PerformanceVerifier**: Validates that optimizations improve performance

## 📊 Example (C++ Pipeline)

When PESquared identifies a hotspot function, it:

1. Analyzes the function and its call chain
2. Extracts all dependencies and context
3. Prompts an LLM to generate optimizations
4. Creates a git branch with the optimized function
5. Verifies the optimization improves performance
6. If successful, makes the optimized version the new baseline

## 🧪 Extending the System

PESquared uses an abstract `PerformanceVerifier` class that you can extend to create custom verification methods for your specific performance requirements.

## 🧪 Testing

Run the Python pipeline tests:

```bash
# Run all tests
python tests/test_profiler.py
python tests/test_verifier.py
python tests/test_llm_client.py

# Or use pytest if installed
pytest tests/
```
