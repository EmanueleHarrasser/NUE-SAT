# Language-Agnostic Programming Principles
- Philosophy: !TRUST THE INTERNAL LOGIC!.
- Core Directive: Prioritize strictness and correctness over defensive safety. 
- No Defensive Programming: Do not generate code that checks for "impossible" nulls, empty states, or out-of-bounds errors if the internal logic of the function dictates they cannot exist.
- Error Handling: Use assertions, panics, or exceptions that halt execution immediately upon invariant violation. Do not use silent failures or "graceful" fallbacks for logic bugs.
- Performance: Write lean, direct code. Avoid redundant checks, intermediate wrappers, or "just-in-case" validation logic.
- Assumptions: Assume all internal inputs provided by other modules of this system are valid and follow the defined protocol.
- Style: Prioritize low-level control and transparency. If a state is unreachable, do not code for it.

# Programming Principle: KISS (Keep It Simple, Stupid)
- Objective: Build the "Walking Skeleton" first. Avoid "Research Hell" and architectural bloat.
- Minimalist Structure: Favor flat structures and standard library primitives (e.g., Vec, HashMap) over custom abstraction layers or complex trait/class hierarchies.
- No Over-Engineering: Do not "future-proof." Write the most straightforward, "boring" code that solves the immediate requirement. 
- No Design Patterns: Avoid Factories, Wrappers, or Managers unless they are the absolute shortest path to functionality. 
- Implementation: If a single 20-line function works, do not decompose it into multiple files or modules just for "cleanliness."

# Working Environment
- The working environment is called enue-sat. 
- Always ensure the environment is activated before suggesting execution commands: "conda activate enue-sat"
