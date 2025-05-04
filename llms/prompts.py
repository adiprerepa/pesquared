from langchain.prompts import PromptTemplate

ALGORITHM_OUTLINE_PROMPT = PromptTemplate(
    input_variables=["function_name", "code"],
    template="""
You are an algorithms analyzer. Analyze the algorithm for the following C++ function {function_name}.

```cpp
{code}
```

Write a detailed, step by step outline of the algorithm as a numbered list.
"""
)

ALGORITHM_BOTTLENECK_PROMPT = PromptTemplate(
    input_variables=["function_name", "code", "steps"],
    template="""
You are a performance engineer. Analyze the following algorithm {function_name} for bottlenecks.

{steps}

We will optimize this step by step.

Structure your response as follows:

# Bottleneck Insights
<List out the function's top 3 bottleneck steps IN REVERSE (don't worry about how to fix just yet)> <the bottlenecks should be MEMBERS of `{function_name}` and NOT `{function_name}` itself>
## Bottleneck `NAME`:
<1 sentence description of step `NAME`>
### Dependants
Fill in the following VERBATIM (filling in for the `variables` by their semantic meaning):
The outputs from `NAME` are DIRECTLY utilized by the future steps 
- `future step 1`
- `future step 2`
-  ...
### Semantic Union to Find Commonly Held Value (X)
First write the following: "I will now perform the semantic union: `future step 1` \cup `future step 2` \cup ... = *`X`*"
`NAME` is valuable because it provides the valuable information of *`X`* to [`future step 1`, `future step 2`, ...].
### Thinking from the Dependants Perspective
Fill in the following VERBATIM (ignoring all context in this repo):
'if I JUST needed *`X`* information and NOTHING else, and I knew NOTHING of this codebase, and *`X`* was ALL I needed, the FASTEST and MOST NAIVE way to get *`X`* would be to perform *`Y`*'.
#### Comparison of *`Y`* and *`NAME`*
Fill in the following: "Question: is *`Y`* is roughly the same as *`NAME`*? Answer: `YES/NO`"
If NO: then write the following:
#### NOTE to SELF
'Consider changing *`NAME`* to *`Y`*'
Else:
#### Nothing here to optimize
N/A

## Do the same for the other 2 bottlenecks
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