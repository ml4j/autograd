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

The repository contains standalone build infrastructure, the documented `Value`,
`ValueNode`, `Accumulatable`, `GradNode`, and `AutogradValue` interfaces, and the immutable scalar implementation
`FloatValue` with tests.
`ValueNodeImpl` implements the computation-node contract with focused tests.

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

## Implementation: `org.ml4j.autograd.scalar.FloatValue`

- A final class implementing `Value<Float, Void>`, constructed with a primitive float.
- Stores fixed scalar data; its supplier returns that data without a mutation API.
- Uses null context because the scalar needs no additional metadata.
- Accepts all primitive float values, including signed zero, infinities, and NaN.
- Tests demonstrate stable repeated data access, independent values, absent context,
  and preservation of floating-point boundary values.

## Agreed interface: `org.ml4j.autograd.node.ValueNode<V>`

Purpose: represent a forward value, its immediate computation dependencies, and its
local backward callback. The type bound is `V extends Value<?, ?>`; no gradient
interface or implementation is required by this contract.

```java
V value();
List<? extends ValueNode<?>> inputs();
void setBackwardFunction(Consumer<V> resultValueBackwardFunction);
Consumer<V> backwardFunction();
void backward();
```

- `value()` returns a non-null represented value without evaluating numerical data.
- `inputs()` returns an unmodifiable, non-null list of non-null nodes in operand order.
  Leaves have no inputs. Duplicate operand references are preserved. Inputs may use
  different numerical data and context types.
- The callback consumes the represented forward/result value, not its gradient.
  Gradient lookup and contribution calculation are responsibilities of the callback.
- Installing a callback replaces the previous one without executing either callback.
  Passing null removes it. `backwardFunction()` returns the installed callback or null,
  without executing it or evaluating data.
- `backward()` invokes the installed callback once with `value()`, or does nothing
  when no callback exists. It does not traverse inputs, seed gradients, or extract
  gradients. A separate coordinator ensures contributions are ready before invocation.
- Callback exceptions propagate; this local operation offers no rollback guarantee.
- Scalar and tensor operations use the same node contract; context such as tensor
  shape remains on the represented value.

## Agreed interface: `org.ml4j.autograd.Accumulatable<V>`

Purpose: describe how values combine into a sum, independently of the state that
stores accumulated gradients. For example, scalar contributions of 2 and 3 sum
to 5; tensor contributions add compatible numerical contents.

```java
V add(V other);
```

- Accepts a non-null compatible operand and returns a non-null sum.
- Addition does not mutate either operand's numerical data, computation graph,
  or gradient state. Storing the sum in an accumulator is a separate responsibility.
- Differentiable implementations retain the operand dependencies and backward rule
  on the result and combine both operands' gradient requirements.
- Numerical data and context must be compatible. Tensor shape compatibility and
  broadcasting rules are defined by concrete numerical implementations, not this interface.
- No result object-identity, evaluation, or caching strategy is prescribed.
- The exception policy for invalid operands is not specified by this interface.
- `FloatValue` remains an implementation of `Value` alone in this increment.

## Agreed interface: `org.ml4j.autograd.node.GradNode<V>`

Purpose: hold the current gradient and combine contributions using the gradient
value's own addition. The type bound is `V extends Value<?, ?> & Accumulatable<V>`.
This supports scalar and tensor gradient values without requiring a complete
autograd-value interface.

```java
Optional<V> gradient();
void accumulate(V contribution);
void reset();
```

- `gradient()` returns a non-null optional without evaluating data. Absence differs
  from a present gradient whose contents are zero. Gradient data/context must be
  compatible with the forward value it differentiates.
- The first contribution is stored directly. Subsequent contributions replace the
  held reference with `current.add(contribution)`, without mutating either operand.
- Contributions and sums are non-null and compatible with the represented gradient.
  Null contributions or sums cause `NullPointerException`. If addition throws or
  returns null, the previously held reference remains unchanged. Compatibility rules
  and their exception types belong to the numerical implementation.
- The node does not directly evaluate data; addition determines any evaluation and
  whether differentiable dependencies are retained.
- Accumulation and reset preserve the node as a stable handle. Held gradient value
  identity is not guaranteed; earlier expressions are not refreshed or mutated.
  Their data stability continues to follow the `Value` contract.
- `reset()` clears only the held reference and is idempotent. It does not release
  forward data or mutate earlier gradient values or their computation graphs.
- The accumulator has no graph-input or backward-execution responsibilities. A
  differentiable gradient value can have its own computation node.

## Agreed interface: `org.ml4j.autograd.AutogradValue<V, D, C>`

Purpose: connect numerical values to their computation and gradient nodes and provide
graph-wide backward entry points. Extends `Value<D, C>` and `Accumulatable<V>`, with
the self-type bound `V extends AutogradValue<V, D, C>`. Gradients use the same concrete
value type, so gradient expressions can have their own computation graphs.

```java
ValueNode<V> valueNode();
GradNode<V> gradNode();
boolean requiresGrad();
void backward(V gradient);
void backward();
```

- `valueNode()` returns a non-null node representing this value and its inputs.
- `gradNode()` returns the same non-null gradient node throughout the value's lifetime,
  including reset and backward passes. Stored gradient values may change; previously
  returned expressions are neither refreshed nor mutated by replacement.
- `requiresGrad()` requests gradient calculation, independently of gradient availability.
  Constants may have computation nodes, but receive no contributions from a backward pass.
- `backward(gradient)` takes a non-null seed compatible with the result's data/context.
  Tensor seed compatibility follows the numerical implementation's rules.
- Traversal visits reachable nodes once by identity, running local callbacks after all
  downstream contributions arrive. Repeated operand positions contribute separately.
- Each pass propagates only new contributions. Leaves are nodes with no inputs; their
  gradients accumulate across passes. Intermediate gradients are replaced. Callers reset
  leaf gradient nodes explicitly to discard previous contributions.
- Gradient computations retain differentiable dependencies for higher derivatives.
  A constant derivative expression need not itself require gradients.
- A null explicit seed causes `NullPointerException`. A root not requiring gradients,
  or a cyclic graph, causes `IllegalStateException`. Callback failures can leave partial
  gradient updates; no graph-wide rollback is guaranteed.
- Passes execute sequentially; concurrent execution over shared values is unsupported.
- `backward()` uses the numerical type's implicit seed (one for float scalars).
  Tensor seed policies are type-specific. Types with no implicit seed throw
  `UnsupportedOperationException` and require explicit seeds.
- Inherited addition follows the existing `Accumulatable` contract; inherited data
  access and context stability follow the existing `Value` contract.

## Implementation: `org.ml4j.autograd.node.impl.ValueNodeImpl<V>`

- A final implementation accepting a deferred represented-value supplier and an input list.
  Its leaf constructor supplies an empty list; neither constructor resolves the value.
- Stores an immutable copy of inputs, preserving order and repeated operand references.
- Rejects null suppliers, input lists, and input nodes at construction; rejects null
  represented values when accessed. The supplier must consistently represent this node's value.
- Initially has no callback. Installation, replacement, and removal do not execute callbacks.
- Local backward execution passes the represented value, not its gradient, and never
  traverses inputs. Callback exceptions propagate without rollback.
- Intended for sequential use; no thread-safety guarantee is provided.
- Existing interface contracts remain unchanged. Tests cover deferred access, data
  non-evaluation, input copying, callbacks, local execution, and failure behaviour.

## Decisions still to finalise

Concrete operation rules, tensor shape/seed policies, and numerical compatibility
validation remain to be specified as implementations are introduced. Earlier
proof-of-concept implementations are reference material, not code to copy wholesale.
