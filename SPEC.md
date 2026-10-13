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
`ValueNode`, `Accumulatable`, `GradNode`, `AutogradValue`, and `BinaryGradient` interfaces,
and the immutable scalar implementation
`FloatValue` with tests.
`ValueNodeImpl` implements the computation-node contract with focused tests.
`GradNodeImpl` implements gradient-state storage, accumulation, and reset with focused tests.
`AutogradValueImpl` and `FloatAutogradValue` provide lazy addition and multiplication,
local propagation, and graph-wide backward execution. Tests and the scalar demo
validate first derivatives, retained gradient expressions, and Hessian-vector products.

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
- Numerical in-place operations are not supported on autograd values. Gradient accumulation
  and reset remain supported mutations of bookkeeping, without changing numerical contents.
  Direct backing-storage changes and changing external data suppliers are unsupported.

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

## Current backward policy: derivative recording and graph retention

Backward produces differentiable gradient expressions and keeps computation graphs
available for reuse. For example, differentiating `x*x*x` produces an expression for
`3*x*x` that can be differentiated again. Tensor applications can use this behaviour
for Hessian-vector products, provided their operation rules support higher derivatives.

- `createGraph` means recording derivative computations so their results retain
  differentiable connections to the original operands. The current policy is broadly
  equivalent to `createGraph = true`; there is no flag or detached-gradient mode.
  Constant derivatives need not require gradients, and each numerical operation must
  supply suitable differentiable rules for the derivative orders it supports.
- `keepGraph` (also called `retainGraph`) means preserving the original graph's
  saved backward state for another pass. The current implementation does not release
  inputs, callbacks, or numerical suppliers after backward, so the graph remains reusable
  while reachable. There is no retention flag or explicit saved-state release policy;
  unreachable objects remain eligible for normal Java garbage collection.
- These are separate concepts: retaining the original graph alone does not make
  computed gradients differentiable. Recording derivatives can also retain references
  to original operands and their graphs, increasing memory use.
- Repeated passes follow the existing accumulation contract: leaf gradients accumulate
  and intermediate gradients are replaced. `gradNode().reset()` clears the stored
  gradient reference; it does not detach previously returned expressions or release
  forward graph state. Neither concept changes the lazy forward-evaluation policy.

Potential future configuration is deferred, not agreed or implemented. Revisit it when:

- First-order training needs numerical gradients without building derivative graphs.
- Larger tensor workloads or long-lived gradient expressions reveal memory pressure
  from retained dependencies, callbacks, suppliers, or future saved tensors.
- Operations introduce saved backward buffers that need an explicit release lifecycle,
  requiring a decision about repeated backward passes after release.
- A new numerical backend or operation supports only first derivatives, requiring
  explicit capability documentation and a policy for unsupported higher derivatives.

Before adding flags, agree their defaults, detachment semantics, release behaviour,
and failure rules. Update interface contracts, Javadoc, and tests together, including
first derivatives, higher derivatives, repeated passes, and reset behaviour.

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

## Implementation: `org.ml4j.autograd.node.impl.GradNodeImpl<V>`

- A final implementation with an initially absent gradient and no constructor-time evaluation.
- Holds the first contribution directly; combines later contributions through `current.add(contribution)`.
- Validates the contribution and sum before replacing state. Addition failures and null
  sums preserve the previous reference, relying on the non-mutating addition contract.
- Reading gradient state and resetting do not evaluate data. Addition decides any evaluation.
- Reset clears only the reference, preserving earlier values and the node handle.
- Intended for sequential use; no thread-safety guarantee is provided.
- Tested with a small immutable accumulatable fixture; `FloatValue` and interface contracts
  remain unchanged. No additional dependencies were introduced.

## Implementation: generic autograd structure and float demonstration

- `org.ml4j.autograd.impl.AutogradValueImpl<V, D, C>` owns stable value and gradient
  nodes, numerical data access, context, and gradient requirements.
- Its value node uses a deferred `self()` reference; no subclass hook is invoked
  during base construction. Subclasses consistently return their concrete instance.
- Generic binary construction delegates numerical operations, context mapping, and result
  construction to concrete hooks. Context is mapped once; data evaluation is deferred.
- Results preserve ordered operand nodes, repeated operands, and the combined
  gradient requirement. Construction does not mutate operand graphs or gradient state.
- Supplied data/context must meet the stable `Value` contract. Null numerical results
  are rejected when evaluated; the base implementation introduces no caching.
- `org.ml4j.autograd.scalar.FloatAutogradValue` is a separate final scalar subclass,
  supporting fixed or deferred stable float data with Void context. `FloatValue` is unchanged.
- `org.ml4j.autograd.scalar.demo.FloatAutogradValueDemo` demonstrates forward addition
  and graph/gradient-state inspection, followed by a graph-wide backward pass.
- Graph-wide backward execution and generic binary derivative rules are implemented as described below.

## Implementation increment: local addition propagation

- Addition installs a callback receiving the forward result value. It reads the incoming
  gradient from that result's gradient node and contributes it unchanged to each operand
  requiring gradients. No result gradient or a constant result causes no propagation.
- Duplicate operand positions contribute separately; accumulation uses the tested
  `GradNode` implementation without mutating shared seeds or earlier gradient expressions.
- The callback does not seed gradients or traverse inputs. Tests seed result gradient
  nodes explicitly and invoke only `valueNode().backward()`.
- Generic binary construction installs a shared callback using derivative rules supplied by the operation.
- This unchanged-gradient rule is demonstrated for scalar addition. Future tensor operations
  involving broadcasting must supply appropriate gradient-shape adaptations explicitly.
- Existing interface contracts and dependencies are unchanged.

## Implementation increment: graph-wide addition backward passes

- Traversal is collected before gradient state changes, using dependency-first ordering
  and identity-based visitation. Cycles are rejected; graph inputs must represent autograd values.
- Each pass saves earlier leaf gradient references and clears reachable gradient state,
  seeds the output, then executes callbacks in reverse dependency order. Only tracked
  nodes with an incoming gradient run their callback.
- Previous leaf contributions are added back only after propagation, so they cannot be
  propagated again. Intermediate gradients belong only to the current pass. Saved expressions
  are not mutated; typed restoration actions avoid unchecked casts for gradient accumulation.
- The base implementation supports explicit seeds and provides an overridable implicit-seed
  hook. Types without an implicit seed require explicit seeds. The float subclass supplies one.
- No numerical data is changed by backward execution. Callback failures can leave partially
  updated state, as allowed by the existing non-transactional backward contract.
- The demo now runs a scalar addition backward pass. Tests cover shared nodes, repeated
  operands, seeds, constants, resets, retained expressions, repeated passes, and cyclic graphs.

## Agreed interface: `org.ml4j.autograd.operators.BinaryGradient<V>`

Purpose: calculate one operand's contribution from a result's incoming gradient
and its forward operands. Scalar addition returns the incoming gradient unchanged;
tensor rules can adapt contributions to operand shapes.

```java
V apply(V incomingGradient, V leftOperand, V rightOperand);
```

- A functional interface with explicit non-null incoming gradient and forward operands
  in their original order, including duplicate references. The forward result itself
  is not passed to this rule.
- Returns a non-null contribution compatible with the differentiated operand's data/context.
- Does not mutate arguments or accumulate state. The caller performs accumulation.
- Differentiable rules retain dependencies for higher derivatives; a constant derivative
  expression need not require gradients itself.
- Numerical compatibility and invalid-argument exception policies belong to concrete rules.

## Implementation increment: shared binary backward callbacks

- `applyBinaryOperator` takes left/right `BinaryGradient` rules and installs the result-value callback.
- The callback skips constant results and absent incoming gradients, invoking only rules
  for tracked operands. It passes the same incoming gradient and ordered operands to each rule.
- Returned contributions are accumulated separately for both operand positions, including
  duplicates. Addition supplies two unchanged-gradient rules; its behaviour is preserved.
- No multiplication operation or dependencies are added in this increment.

## Implementation increment: scalar multiplication and higher derivatives

- `FloatAutogradValue.mul(FloatAutogradValue other)` returns a lazy product, with
  ordered input nodes and the combined gradient requirement. Null operands are rejected.
- The left contribution is `incomingGradient.mul(rightOperand)`; the right contribution
  is `incomingGradient.mul(leftOperand)`. The shared binary callback handles accumulation,
  constants, and duplicate operands. Forward data and earlier expressions are not mutated.
- Derivative rules use differentiable value operations so their expressions retain
  dependencies for higher derivatives. The default scalar seed remains one.
- Tests cover seeds, constant positions, zero derivatives, repeated operands, shared
  branches, repeated passes, reset, and differentiation of retained gradient expressions.
- For `z = x*x + y*x + y*y` at `(0.5, 0.6)`, first derivatives are `(1.6, 1.7)` and
  the Hessian-vector product for `(2, 1)` is `(5, 4)`. Without leaf reset, stored totals
  are `(6.6, 5.7)`. The scalar demo demonstrates the reset version.
- Existing interfaces and dependencies are unchanged. Tensor multiplication remains
  a separate numerical implementation decision, not inferred from scalar multiplication.

## User-introduced arithmetic contracts and wrapper demonstration

- `org.ml4j.autograd.arithmetic.ArithmeticOperations<V extends ArithmeticOperations<?>>`
  defines `V add(V other)` and `V mul(V other)`: non-null compatible operands produce
  non-null results without operand mutation. Numerical types define multiplication
  semantics and compatibility. No inline operations are provided.
- `DifferentiableArithmeticOperations<V, D, C>` extends `AutogradValue` and
  `ArithmeticOperations`, requiring wrapped `D` to implement arithmetic and `V` to
  support the differentiable arithmetic interface. Its default add uses `D::add` and
  two unchanged-gradient rules. Default mul uses `D::mul` and incoming-gradient times
  the opposite operand for each contribution.
- `mapContext(leftContext, rightContext)` and `applyBinaryOperator(other, forward,
  leftGradient, rightGradient, contextMapper)` are public integration hooks for interface
  defaults. The base class provides final binary graph construction and traversal,
  but no longer defines value-level addition. Raw Float implementations supply their own add.
- Unchanged-gradient addition is appropriate for scalars and shape-compatible elementwise
  tensor addition. Broadcasting reductions and non-elementwise tensor product rules need
  explicit adaptations; these defaults do not establish those future semantics.
- `FloatWrapper` extends arithmetic with `float getFloat()`. That user-introduced accessor
  name is preserved in this increment; a change to accessor naming requires explicit agreement.
- `FloatWrapperImpl` stores fixed primitive data and supplies functional numerical arithmetic.
  `FloatWrapperAutogradValue` combines wrapper data with differentiable arithmetic and Void context.
  Its implementation inherits arithmetic defaults and supplies result construction and the default scalar seed.
- The wrapped demo unwraps numerical data with getFloat() and validates the same first-
  and second-derivative examples as the raw Float demo. Wrapper tests cover numerical
  arithmetic, default addition, multiplication, and Hessian-vector products.
- The base class has no numerical addData hook. Raw Float implementations pass numerical
  arithmetic directly to applyBinaryOperator; wrapped values use shared arithmetic defaults.
- The default product rule applies to scalars and compatible elementwise tensor operations.
  Non-elementwise products and broadcast reductions require overrides.

Concrete operation rules, tensor shape/seed policies, and numerical compatibility
validation remain to be specified as implementations are introduced. Earlier
proof-of-concept implementations are reference material, not code to copy wholesale.

## Decisions still to finalise
