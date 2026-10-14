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
`ValueNode`, `Accumulatable`, `GradNode`, `AutogradValue`, `BinaryGradient`, and `UnaryGradient` interfaces,
and the immutable scalar implementation
`FloatValue` with tests.
`ValueNodeImpl` implements the computation-node contract with focused tests.
`GradNodeImpl` implements gradient-state storage, accumulation, and reset with focused tests.
`AutogradValueImpl` and `FloatAutogradValue` provide lazy addition, subtraction,
multiplication, division, and negation,
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

## Agreed interface: `org.ml4j.autograd.DataSupplier<D>`

Purpose: expose data access independently of context, arithmetic, and autograd state.
For example, supply Float scalar data or a backend tensor object.

```java
Supplier<D> data();
```

- Returns a non-null supplier without evaluating data; evaluation is deferred to get().
- Caching, supplier/result identity, data stability, and result nullability are
  implementation-defined. Evaluation failures may propagate from the supplier.
- Adds no context, graph, gradient, or mutation methods.
- Value<D, C> extends DataSupplier<D> and redeclares data() with its existing stronger
  guarantees: stable numerical contents and non-null successful evaluations.
  AutogradValue inherits this capability through Value. Existing implementations
  and the Value data-access contract are unchanged.

## Agreed interface: `org.ml4j.autograd.Value<D, C>`

Purpose: provide numerical data access and the metadata needed to interpret that data.
Extends DataSupplier<D> while preserving the stronger data-access guarantees below.
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
void setRequiresGrad(boolean requiresGrad);
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

### Legacy migration note: `keep_graph` and `create_graph`

This note describes the reference implementation, not the new library's contract.

- Legacy `BackwardConfig.keep_graph` selects whether derivative rules use original
  operands or detached values, controlling differentiable connections for higher derivatives.
- Legacy `AutogradValueProperties.create_graph` is persistent per-value state.
  Backward sets it on the root from `config.keep_graph()`. `addToGrad()` checks
  `requires_grad || create_graph` when deciding whether to accumulate a contribution.
  The property itself does not select the derivative-rule branch.
- Value-conversion constructors copy the property and unary operation results inherit
  it; binary operation results do not explicitly inherit it. It therefore must not
  be interpreted as an independent, consistently propagated derivative-recording flag.
- The new library deliberately has no equivalent persistent property. Its current
  derivative rules retain differentiable dependencies as described above. Any future
  configuration requires a separately agreed backward-pass contract, rather than
  copying the legacy mixture of pass configuration and persistent value state.

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
- `org.ml4j.autograd.demo.FloatAutogradValueDemo` demonstrates forward addition
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
  defaults. The base class provides overridable binary graph construction and graph-wide traversal,
  but no longer defines value-level addition. Raw Float implementations supply their own add.
- Unchanged-gradient addition is appropriate for scalars and shape-compatible elementwise
  tensor addition. Broadcasting reductions and non-elementwise tensor product rules need
  explicit adaptations; these defaults do not establish those future semantics.
- `FloatWrapper` extends arithmetic with `float floatValue()`, following the agreed
  accessor convention. The user explicitly agreed to rename the previous getFloat()
  signature; numerical behaviour is unchanged and no compatibility alias is provided.
- `FloatWrapperImpl` stores fixed primitive data and supplies functional numerical arithmetic.
  `FloatWrapperAutogradValue` combines wrapper data with differentiable arithmetic and Void context.
  Its implementation inherits arithmetic defaults and supplies result construction and the default scalar seed.
- The wrapped demo unwraps numerical data with floatValue() and validates the same first-
  and second-derivative examples as the raw Float demo. Wrapper tests cover numerical
  arithmetic, default addition, multiplication, and Hessian-vector products.
- The base class has no numerical addData hook. Raw Float implementations pass numerical
  arithmetic directly to applyBinaryOperator; wrapped values use shared arithmetic defaults.
- The default product rule applies to scalars and compatible elementwise tensor operations.
  Non-elementwise products and broadcast reductions require overrides.

Concrete operation rules, tensor shape/seed policies, and numerical compatibility
validation remain to be specified as implementations are introduced. Earlier
proof-of-concept implementations are reference material, not code to copy wholesale.

## Agreed increment: unary negation

- `ArithmeticOperations<V>` adds `V neg()`: a non-null arithmetic negation without
  operand mutation. Scalar 2 becomes -2; tensors negate each element without changing shape.
- `DifferentiableArithmeticOperations` supplies default negation through numerical
  `D::neg`, the derivative rule `incomingGradient.neg()`, and unchanged context.
- Public `applyUnaryOperator(UnaryOperator<D> forward, UnaryGradient<V> gradient,
  UnaryOperator<C> contextMapper)` is an integration hook implemented by the base class.
  It rejects null arguments, maps context once without evaluating data, constructs a
  lazy result recording exactly one input, and preserves the operand's requiresGrad.
- `UnaryGradient<V>` defines `V apply(V incomingGradient, V operand)`. It receives
  the non-null incoming gradient and original forward operand, not the forward result;
  returns a non-null compatible contribution without mutation or accumulation; and
  retains differentiable dependencies where needed for higher derivatives.
- The result's local callback reads its gradient and invokes the rule only for a
  tracked operand with an incoming contribution. It accumulates without traversal;
  existing backward coordination, reset, and graph policies apply unchanged.
- Raw Float and wrapped Float implementations demonstrate negation. Constant derivative
  expressions need not require gradients. Negation composed with multiplication can
  preserve nonconstant derivative expressions for second derivatives.

## Agreed increment: composed subtraction

- `ArithmeticOperations<V>` adds `V sub(V other)`: a non-null difference for a
  non-null compatible right operand, without changing operand data, graphs, or gradients.
  Scalar 5 minus 3 is 2; tensors subtract compatible elements under their numerical rules.
- `DifferentiableArithmeticOperations.sub` composes `add(other.neg())`. It constructs
  a negation node for the right operand and an addition node for the result; it does
  not call the wrapped data's sub method or introduce another graph helper.
- Evaluation remains lazy. Context mapping and gradient requirements follow existing
  negation/addition contracts. For scalars or compatible same-shape tensors, left and
  right contributions are respectively the incoming gradient and its negation.
  Broadcasting support depends on appropriate addition overrides.
- Raw Float subtraction uses the same composition; wrapped numerical data provides
  ordinary float subtraction. Constants, repeated passes, accumulation, and higher
  derivatives follow the existing contracts. Null operands cause NullPointerException.
- Direct single-node subtraction is deferred; tensor performance may motivate a
  separately agreed implementation change later.

## Agreed scope: arithmetic operations

Implement the functionality of every method declared in the legacy
`org.ml4j.autograd.arithmetic.operations.ArithmeticOperations<V>` except the
in-place methods `add_(V)` and `sub_(V)`. This is twelve methods in total:

| Legacy method | Purpose | New module status |
| --- | --- | --- |
| `float[] getDataAsFloatArray()` | Read numerical contents as a float array | Provisional `dataAsFloatArray()`; unimplemented |
| `V add(V other)` | Add another value | Implemented |
| `V add(float other)` | Add a scalar constant | Provisional signature; unimplemented |
| `V sub(V other)` | Subtract another value | Implemented |
| `V sub(float other)` | Subtract a scalar constant | Provisional signature; unimplemented |
| `V mul(V other)` | Multiply by another value | Implemented |
| `V mul(float other)` | Multiply by a scalar constant | Provisional signature; unimplemented |
| `V div(V other)` | Divide by another value | Implemented |
| `V div(float other)` | Divide by a scalar constant | Provisional signature; unimplemented |
| `V neg()` | Negate a value | Implemented |
| `V gt(float value)` | Compare with a scalar threshold: greater than | Provisional signature; unimplemented |
| `V gte(float value)` | Compare with a scalar threshold: greater than or equal | Provisional signature; unimplemented |

The pending signatures are now exposed for cross-project contract review, with
Javadoc and throwing defaults only. Final semantics, implementation, and meaningful
behavioural tests remain separate increments.
All operations preserve operand numerical contents; graph/gradient bookkeeping
continues to follow the existing contracts. The legacy implementation remains reference
material and is not modified by these increments.

Before implementing the provisional methods, settle the following details:

- Float-array access: `dataAsFloatArray()` follows the `x()` accessor convention;
  its name is provisional for integration review. Settle copy/alias semantics, tensor flattening order, and
  conversion behaviour for future non-float data. Reading is not differentiation.
- Scalar overloads: define constant-value construction and tensor shape/context
  handling without accidentally requesting gradients for the supplied scalar.
- Comparisons: define the returned value representation, tensor elementwise behaviour,
  and differentiation policy, including thresholds and non-finite inputs. Returning
  `V` in the legacy API does not by itself establish a differentiable comparison.

Tensor broadcasting, numerical compatibility, and backend-specific behaviour must
be agreed as the relevant implementations are introduced. This scope does not add
other legacy tensor operations such as slicing or concatenation.

## Agreed increment: value division

- `ArithmeticOperations<V>` adds `V div(V other)`, returning a non-null quotient
  for a non-null compatible denominator without operand mutation. Numerical types
  define compatibility and zero/non-finite behaviour; float implementations use Java
  IEEE floating-point division, with infinities/NaN rather than zero-division exceptions.
- `DifferentiableArithmeticOperations.div` uses `D::div` through the existing binary
  helper. For result left/right and incoming gradient g, contributions are g/right
  and -(g*left)/(right*right), built with differentiable operations.
- The default supports scalars and compatible same-shape elementwise tensors, not
  matrix division or broadcast-gradient reduction. Such tensor types must override it.
- Results are lazy, record ordered operands including duplicates, and combine gradient
  requirements. Existing context mapping, accumulation, repeated-pass, and higher-
  derivative policies apply. Null operands cause NullPointerException.
- Float derivatives at singularities and floating-point overflow/underflow follow
  the arithmetic of the derivative expressions; no regularisation or finite-value
  guarantee is provided. Analytical derivative examples assume nonsingular inputs.
- Raw Float values implement the same rules; wrapped float data provides numerical
  division. Float overloads are not introduced in this increment.

## Decisions still to finalise

## Provisional arithmetic signatures for integration review

At the user's request, ArithmeticOperations exposes the seven remaining signatures:

```java
float[] dataAsFloatArray();
V add(float other);
V sub(float other);
V mul(float other);
V div(float other);
V gt(float value);
V gte(float value);
```

These are default methods throwing UnsupportedOperationException without evaluating
numerical data or changing graph/gradient state. This keeps existing implementations
buildable while external adapters can compile calls or override the proposed methods.
DifferentiableArithmeticOperations inherits all seven methods; it supplies no new
forward or derivative implementation. Scalar arguments represent constants, not
independently tracked operands. Comparison return type V does not promise differentiability.
Array ownership/conversion, scalar construction/broadcasting, and comparison policies
remain unresolved as listed above. Existing working arithmetic defaults are unchanged.

## Agreed extension: overridable operator construction

At the user's request, AutogradValueImpl.applyBinaryOperator and applyUnaryOperator
are non-final. Subclasses can wrap forward operations and derivative rules and delegate
to super, supporting tensor broadcast-gradient reduction and backend recording.
Overrides must preserve lazy evaluation, context mapping once, ordered graph inputs
(including duplicate binary operands), gradient requirements, local accumulation,
constant/absent-gradient handling, non-mutating operands, and higher-derivative
dependencies. Existing signatures and default method bodies are unchanged.
This enables extension; it does not implement tensor broadcasting or native gradients.

## Agreed interface: `org.ml4j.autograd.AutogradValueFactory<V, D, C>`

Purpose: create autograd leaves with an explicit caller-selected gradient requirement.
The self-type bound is V extends AutogradValue<V, D, C>. This is a functional interface.

```java
V create(Supplier<D> data, C context, boolean requiresGrad);
```

- Returns a new non-null leaf without evaluating numerical data or modifying existing
  values/gradients. Inputs are empty, the backward callback is absent, and its gradient
  is initially absent. requiresGrad() matches the supplied boolean.
- Rejects a null supplier with NullPointerException. Supplied data/context follow
  Value's stable, compatible numerical-data contract, including non-null successful
  evaluations and null context only when metadata is unnecessary. Caching is an
  implementation choice; no supplier or numerical-result identity is promised.
- Scalar factories can accept Float data with Void context; tensor factories can
  accept backend numerical data with shape context. No backend dependency is introduced.
- There is no two-argument overload or implicit gradient requirement. Operation-result
  construction with graph inputs remains separate. No concrete factory is added in
  this increment, and tensor-project callers remain to be migrated.

### Tensor migration decision: properties

AutogradValueProperties is not restored. Tensor adapters will use explicit data,
context, requiresGrad, and input-node arguments in constructors/result construction.
Naming and graph flags require separate decisions; they are not silently transferred
into the new factory contract. Registry exclusion is agreed below.

### Tensor migration decision: no core registry

AutogradValueRegistry is not restored in the new core. Backward discovers reachable
values through graph inputs; model parameter enumeration belongs to the model.
Unreachable Java graph objects are eligible for garbage collection, without a
registry holding strong references to them.

The tensor migration must review existing registry enumeration and clearAutogradGraph
calls individually. Release references to completed results and reset stored gradients
when no longer needed; retained gradient expressions can themselves retain forward
graph dependencies. Gradient reset does not release previously returned expressions.
Backend native-resource ownership and cleanup remain unresolved migration work:
Java graph collection alone does not establish safe DJL/ND4J resource release.
This decision adds no graph-release API and makes no tensor-project changes.

## Agreed increment: leaf gradient toggling

AutogradValue adds void setRequiresGrad(boolean requiresGrad), implemented by the
base class. Only leaves (empty graph-input lists) accept calls; operation results
throw IllegalStateException even for unchanged-flag requests. Leaf calls are idempotent
and do not evaluate data, change numerical contents/context/nodes, or clear gradients.
Resetting stored gradients remains explicit. Factory requiresGrad is the initial flag.

Callers toggle before constructing the next graph and must finish backward and any
higher derivatives on prior graphs before toggling involved leaves. Existing results
retain their original flags; no recursive update, graph rebuilding, or outstanding-graph
detection is provided. Backward callbacks read current operand requirements, so reuse
of earlier graphs after toggling their leaves is unsupported. Execution remains sequential.
The setter is overridable for future backend synchronization, subject to these rules.

Native-gradient recording, selection, and suppliers are deferred. Initial tensor
migration uses library derivative rules; backend native-resource ownership remains
separate work. No native-gradient or tensor-project implementation is added here.

## Agreed utility: `org.ml4j.autograd.cache.CachingDataSupplier<D>`

A final optional Supplier<D> implementation, constructed from Supplier<? extends D>.
Construction rejects null with NullPointerException without evaluating data. get()
evaluates lazily and caches the first successful non-null result, returning that same
object thereafter. Source exceptions propagate and null results cause
NullPointerException; both leave the cache empty so later calls retry.

After success the wrapper releases its source closure reference. It exposes no reset,
invalidation, or cache-state methods. Sequential use only; no thread-safety guarantee.
Callers must preserve cached numerical contents to satisfy Value stability. It performs
no native-resource cleanup. Tensor adapters can opt in to cache random generation or
native-array allocation; the core base class still leaves caching optional. This class
implements Supplier, not the DataSupplier owner capability. The legacy clearCache()
contract is not restored. Tensor integration remains a separate migration increment.

## Agreed interface: `org.ml4j.autograd.Parameter<V>`

Purpose: hold a replaceable current autograd leaf through a stable parameter handle.
The bound is V extends AutogradValue<V, ?, ?>.

```java
V value();
void replace(V value);
```

- value() returns the non-null current leaf without numerical evaluation, with the
  same identity until replacement. Leaves have empty graph-input lists.
- replace selects the supplied leaf for future reads without numerical evaluation.
  Null causes NullPointerException and graph inputs cause IllegalArgumentException;
  invalid replacement preserves the current reference. Replacing with the current
  leaf is permitted. Neither old nor new leaf's data, context, graph, gradient flag,
  or stored gradient is modified. Fresh training steps should supply fresh leaves.
- Callers provide values compatible with the model's numerical type and context
  (including tensor shape); compatibility validation is implementation-defined.
- Parameter is the mutable selection handle; Value numerical contents remain stable
  for life. Forward graphs capture returned leaves rather than live parameter reads,
  so previously built graphs retain old values after replacement.
- Use sequentially. No arithmetic, optimizer, gradient storage, native-resource
  cleanup, or concurrent replacement/graph-construction contract is introduced.
- This increment introduces only the interface and documentation. Concrete handle
  implementation and tensor/GPT migration remain separate increments.

## Implementation: `org.ml4j.autograd.impl.ParameterImpl<V>`

Final Parameter implementation constructed from a non-null autograd leaf. Constructor
and replace validate empty graph-input lists without evaluating data. Validation
completes before assignment, so rejected replacement preserves the current reference.
value returns the held leaf; replacement changes only that reference. Existing values,
gradients, and graphs remain unchanged. Model numerical/context compatibility is the
caller's responsibility; no tensor shape validation or native-resource cleanup is added.
Sequential use only. Tensor/GPT integration remains a separate migration increment.

### Tensor migration decision: gradient access and reset

Tensor callers will replace legacy grad() (nullable) with gradNode().gradient()
(Optional), and resetGradient() with gradNode().reset(). No legacy aliases are added
in the core. The gradient node is the stable handle; its stored expression can change.
Previously returned gradient values do not refresh after another pass. Callers read
the handle again for the latest gradient and handle absence explicitly. This records
the migration choice only; tensor callers remain unchanged in this increment.

### Tensor migration decision: implicit backward seed

Tensor adapters will override defaultGradient() to supply a constant leaf containing
ones with the same shape/context as the result and requiresGrad false. This preserves
the tensor project's implicit backward seed policy; backward(seed) remains available
for explicit compatible seeds. Replace the legacy multiplicativeIdentity hook with
this adapter override. The core default still requires an explicit seed unless a
subclass defines one. This records the decision; adapter changes remain deferred.

### Tensor migration decision: backward configuration

BackwardConfig is not restored. Tensor callers will use backward() or backward(seed)
without configuration, following the core's existing derivative-recording and graph-
retention policy. Derivative expressions preserve higher-derivative dependencies where
operation rules support them; graphs remain available for repeated passes. Native-
gradient selection and configurable recording/retention policies remain deferred.
This decision records migration direction; tensor callers are not changed here.

### Tensor migration decision: broadcast-gradient reduction

Preserve the tensor adapter's existing reduction of output-shaped gradient
contributions back to each broadcast operand's original shape. Migrate its
applyBinaryOperator override to BinaryGradient<V>, whose arguments are incoming
gradient, left operand, and right operand, and delegate wrapped rules to super.
The adapter retains its shape checks and sum/squeeze reduction logic; the generic
core provides no automatic tensor broadcasting reduction. Validate scalar/tensor,
leading-axis, and singleton-axis broadcasting during adapter migration. This records
the decision only; no tensor code is changed here.

### Tensor migration decision: scalar arithmetic overloads

Tensor add(float), sub(float), mul(float), and div(float) will use the unary
operator helper with the float captured as a fixed constant. Each operation records
only the tensor input and preserves its shape/context. No scalar leaf is constructed,
no gradient is calculated for the float, and these overloads require no separate
scalar-tensor broadcasting. Value overloads remain available for tracked operands.

Forward evaluation delegates to wrapped numerical scalar operations. Incoming-gradient
contributions are unchanged for add/sub, multiplied by the constant for mul, and
divided by the constant for div. Contributions use differentiable operations to
preserve higher-derivative dependencies. Numerical boundary behaviour follows the
backend arithmetic; no regularisation is introduced. Implement these operations
in the tensor adapter during migration. Existing provisional core defaults still
throw UnsupportedOperationException; this increment records the decision only.

### Tensor migration decision: constant comparison masks

Tensor gt(float) and gte(float) will lazily produce elementwise float masks: 1 where
true and 0 where false, preserving tensor shape/context. Results are constant leaves
with requiresGrad false, empty graph inputs, and no backward callback. The adapter
will construct them directly through createAutogradValue rather than the unary helper,
which inherits the operand's gradient requirement. Capturing an input data supplier
for lazy evaluation does not establish a differentiable graph edge.

Gradients do not flow through comparisons. ReLU can use gt(0f) as a constant
multiplier in its own derivative rule. Higher derivatives follow this fixed-mask
convention; no derivative of the comparison or mathematical smoothness at a threshold
is promised. Finalise/document backend boundary behaviour and validate equality,
non-finite inputs, mask data/shape, constant status, and ReLU propagation during tensor
migration. Provisional core comparison defaults remain unsupported; no tensor code
is changed in this decision increment.

### Tensor migration decision: float-array access ownership

Migrate getDataAsFloatArray() to dataAsFloatArray(). Successful reads return a
non-null float array representing numerical contents in the backend's documented
flattening order. The backend may return a copy or shared storage; copying and
zero-copy access are not guaranteed. Callers must treat the returned array as
read-only, even when a particular backend currently copies. Callers needing editable
storage explicitly copy the array. Java arrays do not enforce this restriction.
Access can evaluate numerical data and require conversion or transfer; it does not
change graph/gradient state. Non-float conversion details and backend flattening
order must be documented during adapter migration. No mandatory copy is introduced.
The provisional default remains unsupported until implemented.

### Tensor migration decision: remove inline numerical operations

Remove add_ and sub_ from tensor adapters/API during migration, consistent with the
core's lifetime-stable numerical contents. Use functional add/sub for arithmetic
and new leaves through Parameter.replace for optimizer updates. Source review found
no direct add_/sub_ production calls in minimal-maven-project; both numerical adapters
still declare the methods and the DL4J adapter test asserts their in-place behaviour.
Remove those obsolete mutation assertions when migrating the adapters, retaining
functional arithmetic validation. Tensor code is unchanged in this decision increment.

### Tensor migration decision: names outside core values

Do not restore name() or name_() on core Value, AutogradValue, or Parameter contracts.
The model stores parameter names separately, for example in Map<String, Parameter<T>>.
Names identify stable handles and remain associated when their current leaves are
replaced. Existing tensor naming usages will be removed or moved to model metadata
during migration. Operation/debug labels are optional future metadata, not part of
this increment or the core numerical-value contract.

### Tensor migration decision: copying deferred

Tensor copying is excluded from the current migration scope. Do not restore core
copy/mapped-value constructors or introduce tensor copy semantics now. Sharing versus
duplicating numerical storage and retaining versus detaching graph connections remain
unresolved. Review existing copy-constructor callers during migration; adapt necessary
callers within supported explicit construction or report blockers, without silently
choosing copy semantics. No tensor constructors or callers are changed here.

## Shared scalar arithmetic and comparison defaults

At the user's request, move the six tensor overrides into the core
DifferentiableArithmeticOperations interface. Scalar add/sub/mul/div use the unary
helper with fixed constants and identity context mapping; contributions are g, g,
g*c, and g/c. gt/gte delegate to wrapped numerical masks using a new public
applyConstantOperator(forward, contextMapper) integration hook, supplied by the base.
The hook maps context once, defers data, and creates an untracked leaf with empty
inputs, absent gradient, and no backward callback. It is overridable subject to these
requirements. Existing data, graph, and gradient state are unchanged.

Tensor adapters inherit all six defaults. Wrapped numerical implementations still
must implement the underlying operations; provisional ArithmeticOperations defaults
remain unsupported. No raw-float or numerical FloatWrapper implementation is added
in this increment. dataAsFloatArray delegation stays in the tensor adapter.
This supersedes the earlier statement that all six differentiable defaults are
unsupported and the tensor-specific construction described in migration decisions.

## Agreed increment: declarations-only numerical arithmetic

ArithmeticOperations contains only abstract method signatures and Javadoc; all seven
throwing defaults are removed. Implementers must provide every declared capability.
DifferentiableArithmeticOperations retains the six shared arithmetic/mask defaults.
Concrete FloatWrapperImpl now supplies the previously missing numerical operations:
scalar add/sub/mul/div use Java float arithmetic; gt/gte use Java comparisons with
zero/one encoding (NaN comparisons false); array access returns a fresh one-element
array. FloatWrapperAutogradValueImpl delegates array access to wrapped data.

This supersedes provisional unsupported-default documentation. Raw FloatAutogradValue
is not an ArithmeticOperations implementer and gains no new methods in this increment.
Tensor backends already implement all numerical signatures and inherit shared
differentiable defaults. Returned arrays remain read-only by the interface contract.

## Agreed activation contracts and ReLU implementation

ActivationOperations<V> is a separate non-mutating numerical capability declaring
V relu(): max(0,x), elementwise and shape-preserving for tensors. Results are non-null.
DifferentiableActivationOperations combines this with differentiable arithmetic;
wrapped D must supply both numerical arithmetic and activations. Its default uses
the existing unary helper, preserves context/requirements, and contributes
incoming.mul(operand.gt(0f)). The mask is constant: the derivative is zero at zero,
zero for negative inputs, and one for positive inputs. Higher derivatives use fixed
masks while retaining incoming-gradient dependencies; no smoothness at zero is claimed.

FloatWrapperImpl implements Math.max(0f,value): signed zeros yield positive zero,
NaN is preserved, and infinities follow Java max. Comparison with NaN yields a zero
mask; derivative arithmetic follows IEEE rules without finite-result guarantees.
FloatWrapperAutogradValue inherits the shared activation default. The micrograd-style
AutogradDemo prints primitive forward/gradient values. No tensor-project changes,
additional activations, native-gradient integration, or dependencies are added.

## Shared mathematical-functions capability

The requested migration introduces org.ml4j.autograd.math.MathOperations<V> and
DifferentiableMathOperations<V, D, C>. Numerical declarations are exp(), log(),
and pow(float exponent), with non-null non-mutating results and elementwise,
shape-preserving tensor semantics. D supplies arithmetic and math; V supplies
differentiable arithmetic and math. Activation capabilities are independent.

Defaults preserve the existing tensor rules exactly: exp contributes g*x.exp(),
log contributes g/x, and pow(p) contributes g*x.pow(p-1)*p. They use one lazy unary
input, identity context mapping, and inherited requiresGrad. Exponents are fixed
float constants; derivative expressions retain graph dependencies for higher orders.
No clipping, domain validation, or p=0/p=1 special handling is introduced. Thus
pow(0) at zero can produce a NaN gradient, and higher polynomial derivatives at zero
can encounter singular intermediates. Singular/non-finite derivatives and floating
exponent subtraction retain ordinary backend arithmetic; no finite-result guarantee
or universal boundary equivalence across numerical backends is promised.

FloatWrapperImpl delegates to Java Math.exp/log/pow with double intermediates and
float rounding, including Java NaN, infinity, signed-zero and domain semantics.
DJL/DL4J numerical implementations remain unchanged. Numerical accuracy follows each
backend, with tolerance-based finite-value tests. Valid smooth-domain derivatives
are tested through second order. Uniform boundary policies and improved power
special cases remain separate decisions requiring agreement.

TensorOperations and Tensor inherit the numerical and differentiable capabilities;
local declarations and exp/log/pow graph rules are removed. Scalar FloatWrapper
and FloatWrapperAutogradValue also adopt these capabilities. Softmax and reductions
remain tensor operations.


## Shared sigmoid integration

ActivationOperations declares non-mutating sigmoid(), applying 1 / (1 + exp(-x))
elementwise with preserved tensor shape. DifferentiableActivationOperations supplies
one lazy unary graph rule, preserving context and requiresGrad, with contribution
incoming * s * (1 - s), where s = operand.sigmoid(). Contributions remain
differentiable for higher derivatives. Constants remain untracked. No clipping or
finite-result guarantee is added; backend rounding/non-finite arithmetic is retained.
FloatWrapperImpl uses a stable sign-dependent double calculation rounded to float:
both signed zeros yield 0.5, infinities yield zero/one, and NaN remains NaN.
TensorOperations inherits the declaration, and tensor wrappers inherit the shared
rule; DJL/DL4J numerical implementations remain in the tensor project unchanged.
Softmax/reduction migration remains separate.


## Scalar package organization

FloatWrapper, FloatWrapperImpl, FloatWrapperAutogradValue, and
FloatWrapperAutogradValueImpl live in org.ml4j.autograd.scalar alongside the raw
float scalar types. They combine arithmetic, activation and math capabilities,
so their package is independent of any one operation family. Both scalar demos
live in org.ml4j.autograd.demo and tests follow the production package.
Consumers must update imports from org.ml4j.autograd.arithmetic.scalar; no
compatibility aliases are supplied. Numerical and graph contracts are unchanged.


## Cache utility package

CachingDataSupplier and its tests live in org.ml4j.autograd.cache. Consumers
must update imports from org.ml4j.autograd.CachingDataSupplier; no compatibility
alias is provided. Lazy evaluation, retry, sequential-use and ownership contracts
remain unchanged.
