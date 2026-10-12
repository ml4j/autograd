# Specification

## Purpose

Build a small Java automatic differentiation library incrementally, starting with
scalar examples and explicit, documented interface contracts.
Keep contracts adaptable to matrices and tensors; scalar examples are the initial
validation scope, not a restriction on numerical data types.

## Constraints

- The repository builds independently using Java 25 and its Maven Wrapper.
- No dependency on legacy code or sibling projects.
- Each unit has one clear responsibility. Implementations have tests demonstrating
  their contracts, including relevant boundary cases.

## Current state

The repository contains standalone build infrastructure and the documented `Value`
interface. No library implementation has been introduced yet.

## Interface decisions and change control

### Javadoc convention

Start type and method Javadoc with short, plain-language statements explaining
the purpose or behaviour. Follow with simple examples where useful, including
scalar and tensor examples when relevant. Present the precise contract details
after this introduction: guarantees, constraints, nullability, identity, lifecycle,
and failure behaviour. Finish with the applicable parameter, return, and exception tags.

The introduction and examples must agree with the detailed contract. Keep examples
illustrative; do not imply dependencies or functionality that the contract does not provide.

### Contract changes

Record the agreed decisions for each new interface here when it is introduced:
its purpose, responsibilities, method signatures, observable behaviour, and relevant
constraints. Distinguish agreed contracts from unresolved questions.

Established interface contracts must not change without the user's explicit agreement.
This includes signatures and semantics, such as mutation, identity, nullability,
lifecycle, and failure behaviour. Propose the reason and impact of a change before
implementing it, then update this specification, Javadoc, and affected tests together
after agreement. Refactoring an implementation must preserve the agreed contract.

## Agreed interface: `org.ml4j.autograd.Value<D, C>`

Purpose: provide numerical data access and the metadata needed to interpret that data.
`D` describes the numerical data type; `C` describes its context type.
For example, `D` can be a tensor implementation type and `C` a shape or size type
describing its dimensions. These examples introduce no tensor-library dependency.

```java
Supplier<D> data();
C context();
```

- `data()` returns a non-null supplier without evaluating numerical data. Successful
  supplier evaluations return non-null data; null does not represent absence.
- Numerical contents and context remain stable throughout the value's lifetime.
  A lazy supplier must represent the same contents on repeated evaluations, rather
  than a changing source. Context must remain compatible with the numerical data.
- Stability is semantic, not an object-identity guarantee. Implementations and callers
  must not mutate exposed data or context objects in ways that change the represented value.
- Caching is an implementation choice. Neither supplier nor result/context object
  identity is guaranteed. Evaluation failures may propagate from the supplier;
  this interface defines no recovery policy or particular evaluation exception type.
- `context()` does not evaluate numerical data. It may return null only when no
  additional metadata is needed; `Void` is the intended context type for such values.
- The contract provides no mutation, cache invalidation, gradient, graph, or naming operations.

## Decisions still to finalise

Review and document graph-node roles, gradient accumulation,
reset, and backward-pass behaviour as their interfaces are introduced. Earlier
proof-of-concept designs are candidates to evaluate, not requirements to copy wholesale.
