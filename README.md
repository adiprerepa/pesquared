# PESquared - Performance Engineering with LLMs

PESquared is an automatic performance engineering system that uses Large Language Models (LLMs) and performance profiling data to intelligently explore and apply optimizations to your codebase.

## 🚀 Features

- **Call Chain Optimization**: Analyzes entire call chains to optimize both leaf functions and their parents
- **Iterative Performance Improvement**: Applies optimizations gradually, guided by measured performance gains
- **Intelligent Optimization Selection**: Uses LLMs to generate optimizations tailored to your specific codebase
- **Automatic Error Correction**: Detects compilation errors and prompts the LLM to fix them automatically
- **Performance Verification**: Validates that optimizations actually improve performance before accepting them
- **Git Integration**: Creates branches for each optimization, making it easy to review and manage changes

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

## 📋 Requirements

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

## 🔍 Usage

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

## 🔐 Environment Setup

Create a `.env` file with your API keys:

```
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
```

## 🛠️ Architecture

PESquared consists of several interconnected components:

- **StackAnalyzer**: Processes stack trace data to identify performance hotspots
- **DependencyExtractor**: Extracts function dependencies and context
- **FunctionOptimizer**: Generates optimized implementations using LLMs
- **PerformanceVerifier**: Validates that optimizations improve performance

## 📊 Example

When PESquared identifies a hotspot function, it:

1. Analyzes the function and its call chain
2. Extracts all dependencies and context
3. Prompts an LLM to generate optimizations
4. Creates a git branch with the optimized function
5. Verifies the optimization improves performance
6. If successful, makes the optimized version the new baseline

## 🧪 Extending the System

PESquared uses an abstract `PerformanceVerifier` class that you can extend to create custom verification methods for your specific performance requirements.
