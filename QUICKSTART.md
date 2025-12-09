# Quick Start Guide - Python Optimization Pipeline

Get started with PESquared's Python optimization pipeline in under 5 minutes!

## Installation

```bash
# Clone the repository
git clone https://github.com/adiprerepa/pesquared.git
cd pesquared

# Install dependencies
pip install -r requirements.txt
```

## Your First Optimization (No API Key Needed!)

### Option 1: Interactive Jupyter Notebook

```bash
jupyter notebook notebooks/01_baseline_profiling_demo.ipynb
```

This notebook demonstrates:
- How to profile Python functions
- Identifying performance bottlenecks
- Manual optimization techniques
- Performance comparison (achieves 12,000x+ speedup on Fibonacci!)

### Option 2: Command-Line Demo

```bash
# Run baseline profiling demo
python scripts/run_demo.py --demo baseline

# Run LLM optimization demo (uses dummy client, no API key needed)
python scripts/run_demo.py --demo llm
```

## Quick Example: Optimize Your Own Function

```python
from optimizers.optimizer import optimize_with_llm
from optimizers.llm_client import create_llm_client

# Define your function
def slow_function(n):
    result = []
    for i in range(n):
        if i % 2 == 0:
            result.append(i ** 2)
    return result

# Create dummy client (works offline)
client = create_llm_client('dummy')

# Optimize!
result = optimize_with_llm(slow_function, 1000, llm_client=client)

print("Explanation:", result.explanation)
print("\nOptimized code:")
print(result.optimized_code)
```

## Using Real LLM (Optional)

To get real optimization suggestions from OpenAI:

```bash
# Set your API key
export OPENAI_API_KEY="your-api-key-here"

# Run with real LLM
python scripts/run_demo.py --demo llm --use-real-llm --model gpt-3.5-turbo
```

Or in Python:

```python
from optimizers.llm_client import create_llm_client
import os

# Set API key
os.environ['OPENAI_API_KEY'] = 'your-key-here'

# Create real OpenAI client
client = create_llm_client('openai', model='gpt-4')

# Use it with optimize_with_llm
result = optimize_with_llm(slow_function, 1000, llm_client=client)
```

## Verify Correctness and Measure Performance

```python
from verifiers.verifier import check_correctness, compare_performance

# Execute the optimized code (⚠️ review it first!)
namespace = {}
exec(result.optimized_code, namespace)
optimized_func = namespace['slow_function']  # or whatever your function name is

# Check correctness
test_cases = [
    ((100,), {}),
    ((1000,), {}),
    ((10000,), {}),
]
passed, errors = check_correctness(slow_function, optimized_func, test_cases)

if passed:
    print("✅ Correctness verified!")
    
    # Measure performance improvement
    perf = compare_performance(slow_function, optimized_func, 10000, n_runs=50)
    print(f"Speedup: {perf['speedup']:.2f}x")
    print(f"Improvement: {perf['improvement_pct']:.1f}%")
else:
    print("❌ Correctness check failed:")
    for error in errors:
        print(error)
```

## Run Tests

```bash
# Run individual test suites
python tests/test_profiler.py
python tests/test_verifier.py
python tests/test_llm_client.py
python tests/test_integration.py

# Or use pytest
pytest tests/
```

## What's Next?

1. **Learn the basics**: Start with `notebooks/01_baseline_profiling_demo.ipynb`
2. **Try LLM optimization**: Explore `notebooks/02_llm_guided_optimization.ipynb`
3. **Experiment**: Optimize your own functions
4. **Read the docs**: Check out `PYTHON_PIPELINE.md` for detailed information

## Demo Commands Reference

```bash
# Show help
python scripts/run_demo.py --help

# Baseline profiling demo
python scripts/run_demo.py --demo baseline

# LLM optimization (offline, dummy client)
python scripts/run_demo.py --demo llm

# LLM optimization (with real OpenAI)
python scripts/run_demo.py --demo llm --use-real-llm --model gpt-4

# Generate prompt for custom function
python scripts/run_demo.py --demo prompt --function my_file.py --name my_function
```

## Important Notes

⚠️ **Safety**: The dummy client is completely safe. When using real LLMs:
- Review generated code before running it
- Use in isolated environments for testing
- Don't run on production systems without thorough review

📊 **Best Results**: Works best with:
- Small, self-contained functions
- Clear algorithmic inefficiencies
- Functions that don't rely on complex external state

🎓 **Learning**: This is a research/demo tool designed for:
- Understanding LLM-guided optimization
- Experimenting with performance engineering
- Educational purposes

## Need Help?

- **Documentation**: See `README.md` and `PYTHON_PIPELINE.md`
- **Examples**: Check the `notebooks/` directory
- **Issues**: Open an issue on GitHub

Happy optimizing! 🚀
