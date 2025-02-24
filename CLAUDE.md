# PESquared Project Commands & Guidelines

## Commands
- **Run main script**: `python main.py <codebase_dir> <stacks_dir> [--debug] [--num-functions N] [--model MODEL] [--provider {openai,anthropic}]`
- **Analyze stacks**: `python analyze_stacks.py <stacks_dir> [--top_n N]`
- **Extract dependencies**: `python extract_deps.py <codebase_dir> <function_name>`
- **Optimize function**: `python optimize_function.py <codebase_dir> <function_name> [--model MODEL]`

## Environment
- Set `OPENAI_API_KEY` and/or `ANTHROPIC_API_KEY` in `.env` file or as environment variables
- Install dependencies: `pip install -r requirements.txt`

## Code Style Guidelines
- **Imports**: Group standard libraries first, then third-party, then local imports
- **Types**: Use Python type hints for functions and classes (`from typing import Dict, List, Optional`)
- **Classes**: Use dataclasses where appropriate. Include docstrings for classes and methods
- **Error Handling**: Use try/except blocks with specific exceptions and error logging
- **Logging**: Use the standard `logging` module instead of print statements
- **Naming**: Use snake_case for functions and variables, PascalCase for classes
- **Documentation**: Include docstrings for all functions with Args/Returns sections

## C++ Performance Optimization
- When optimizing C++ functions, maintain identical signatures and external behavior
- Function optimizations will attempt up to 3 retries if compilation fails or tests fail
- The retry mechanism sends error feedback to the LLM to fix issues automatically

## Project Features
- Automatic performance testing for optimized functions
- Test validation to ensure functional correctness
- Integration with git for branch and commit management