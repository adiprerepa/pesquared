"""Prompt templates for LLM-based code analysis and optimization."""

from langchain.prompts import PromptTemplate

ALGORITHM_BOTTLENECK_PROMPT = PromptTemplate(
    input_variables=["function_name", "code", "bottleneck"],
    template="""
You are acting as a **senior performance engineer**.
The *only* code fragment we will focus on for now is the following call inside
`calcColor(unsigned char*, Autonoma*, Ray, unsigned int)`

```cpp
{bottleneck}
```

The full source for that function is supplied below (enclosed in triple back-ticks) so that you can see how the data produced by {bottleneck} flows into later stages.

```cpp
{code}
```

**Goal:** Identify *bottlenecks*—but **do not** propose fixes yet.
We will optimize iteratively in later steps.

**Deliverable format (strict):**

# Bottleneck Insights

## Bottleneck `NAME`:

<one-sentence summary of what "NAME" is and why it may be costly>

### Dependants

The outputs from **`NAME`** are *directly* consumed by the following later steps (include snippet + semantic role for each):

* `A` - <what the variable/function is and why it needs the result>
* `B`
* …

### Analysis of the Utilization of `NAME`

In the abstract, performing `NAME` is useful because it provides …
Now enumerate **exactly what information** `A`, `B`, … actually read from `NAME`, reasoning step-by-step. Append *θ( … )* or *O( … )* for the cost of each access pattern.

#### Analysis of the Alternatives to `NAME`

For **each** concrete need of `A`, `B`, …, list alternative ways to obtain just that information (not the whole of `NAME`).

1. *ALT₁* - short description (cost: O( … ))
2. *ALT₂* - …

#### Note to Self

1. Consider changing `NAME` to `ALT₁`
2. Consider changing `NAME` to `ALT₂`
3. …

**Mandatory style rules**

1. Preserve the headings and bullet structure exactly.
2. Replace `NAME`, `A`, `B`, etc. with the real identifiers & meanings you infer.
3. No optimization suggestions—only analysis and alternatives listing.
4. Use tight technical language (one idea per sentence; avoid fluff).
5. Cite code lines or variable names verbatim; do **not** paraphrase them.
6. When giving costs, favour tight-bound Big-O or Θ notation.
7. Think **step-by-step** before writing; include *all* intermediate reasoning that affects complexity.

**Begin your analysis below this line.**
"""
)

ARCHITECTURE_BOTTLENECK_PROMPT = PromptTemplate(
    input_variables=["function_name", "code"],
    template="""
You are a C++ performance engineer focusing on architectural and compiler-level optimizations. Analyze the following function {function_name}.

```cpp
{code}
```

Identify potential CPU, memory, instruction pipeline, cache, or SIMD/GPU inefficiencies.
List the top 3 architectural or compiler-related issues with suggestions.
"""
)

OPTIMIZATION_PROMPT = PromptTemplate(
    input_variables=["function_name", "code", "algorithm_bottlenecks", "architecture_bottlenecks"],
    template="""
You are an expert C++ performance engineer. Optimize the following C++ function {function_name} for performance.

Please note:
- The code is correct and compilable as is
- Return the optimized function as a drop-in replacement
- Do NOT return a diff; Do NOT skip lines; return the full optimized function.

```cpp
{code}
```

Your response is machine-processed, so include every detail verbosely. Structure your response as follows:

<Outline your plan for optimizing the function>.
<If no optimizations are possible, state that clearly and end your response here.>

# {function_name}

## New Function
```cpp
<Provide the optimized version of the function here as a drop-in replacement. Only change the body. Keep the same header. Ensure that it is a complete, compilable C++ function.>
```

## New Imports
If new imports needed:
```cpp
#include <<new_imports>>
```
Else:
N/A

# New Flags
If new make flags needed:
```cpp
FLAGS += <new_flags>
```
Else:
N/A

# Branch Name
<git-compatible branch name for your refactor>

# Commit Message
<git-compatible commit message for your refactor>


----------------------------

For your reference, here are some additional notes:
Bottleneck Notes:
{algorithm_bottlenecks}
{architecture_bottlenecks}
"""
)

ERROR_PROMPT = PromptTemplate(
    input_variables=["function_name", "code", "error"],
    template="""
You are a C++ software engineer. Fix the following C++ function {function_name}.
```cpp
{code}
```

Error:
{error}

Structure your response as follows:

# Error Analysis
<Analyze the error in detail. What is it? Why is it happening? What are the implications of this error? How does it affect the function's performance or correctness? Be specific.>

# Fix
<Provide a detailed, step-by-step explanation of how to fix the error. What changes need to be made? Why are these changes necessary? How do they address the error? Be specific.>

# New Function
```cpp
<Provide a fixed version of the function here as a drop-in replacement>
```

## New Imports
If new imports needed:
```cpp
#include <<new_imports>>
```
Else:
N/A

# New Flags
If new make flags needed:
```cpp
FLAGS += <new_flags>
```
Else:
N/A

# Terminal Commands
If we need to run any terminal commnands to fix this error.
```bash
<terminal commands>
```
Else:
N/A
"""
)
