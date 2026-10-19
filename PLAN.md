# Development plan

## Approach

For each increment, define a small interface and its purpose, review its contract,
then add an implementation and tests when requested. Validate the increment before
moving to the next. This plan records direction, not permission to implement all stages.

## Progress

- [x] Standalone Maven setup and wrapper.
- [x] Initial agent instructions, specification, and plan.
- [x] Introduce the numerical value/data-access contract.
- [x] Implement and test basic data access.
- [x] Introduce computation-node and local propagation contracts.
- [x] Introduce the non-mutating `Accumulatable` contract.
- [x] Introduce the gradient-state `GradNode` contract.
- [x] Introduce `AutogradValue` to connect the agreed contracts.
- [x] Implement and test the value-node contract.
- [x] Implement and test the gradient-state contract.
- [x] Implement scalar addition and backward traversal.
- [x] Validate repeated passes, shared inputs, resets, and higher derivatives for scalar addition/multiplication.

## Completed increment: value contract

Introduce `org.ml4j.autograd.Value<D, C>` with the agreed data-access contract.

- Add only the interface, with purpose and method Javadoc.
- Record non-null numerical data, lazy access, stable data/context, optional context,
  and implementation-defined caching in `SPEC.md`.
- Do not add an implementation, tests, or dependencies in this increment.
- Validate with the standalone Maven Wrapper build and check Javadoc against the spec.

Status: complete. Added the interface and matching specification; the standalone
`./mvnw -o verify` build passed. No implementation, tests, or dependencies were added.

## Completed increment: scalar data access

Implement `org.ml4j.autograd.scalar.FloatValue` as a final, immutable implementation
of `Value<Float, Void>` constructed from a primitive float.

- Return a supplier of the stored scalar; return null context because no metadata is needed.
- Document usage and stability without adding mutation, operations, or graph functionality.
- Add JUnit as a test-only dependency and test repeated reads, independent values,
  absent context, and floating-point boundary values.
- Validate the standalone build with `./mvnw verify`.

Status: complete. The standalone `./mvnw -o verify` build passed with four tests.
The agreed `Value` interface contract remains unchanged.

## Completed increment: value-node contract

Introduce only `org.ml4j.autograd.node.ValueNode<V>` before gradient-related interfaces.

- Document the represented value, ordered input nodes, and result-value callback.
- Record the contract in `SPEC.md`, distinguishing one local backward step from graph traversal.
- Add no node implementation or dependencies in this increment.
- Check Javadoc against the specification and run the standalone wrapper build.

Status: complete. The interface and specification agree; the standalone
`./mvnw -o verify` build passed, including the four existing scalar tests.
No node implementation or dependencies were added.

## Completed increment: value-combination contract

Introduce only `org.ml4j.autograd.Accumulatable<V>` with non-mutating addition.

- Document how gradient values combine without managing the stored accumulator state.
- Include scalar and tensor examples and record the contract in `SPEC.md`.
- Add no implementation, dependencies, or changes to `FloatValue` in this increment.
- Check documentation consistency and validate the standalone wrapper build.

Status: complete. Added the interface and matching specification. The standalone
`./mvnw -o verify` build passed with all four existing tests. `FloatValue` is unchanged.

## Completed increment: gradient-state contract

Introduce only `org.ml4j.autograd.node.GradNode<V>`, requiring `Value` and
`Accumulatable` for the held gradient type.

- Document optional gradient access, non-mutating value combination, and reset.
- Record stable-handle and replaceable-expression semantics in `SPEC.md`.
- Add no implementation or dependencies; leave existing interfaces unchanged.
- Check Javadoc against the specification and validate the standalone wrapper build.

Status: complete. Added the interface and matching specification. The standalone
`./mvnw -o verify` build passed, including all four existing tests. No implementation
or dependencies were added.

## Completed increment: autograd-value contract

Introduce only `org.ml4j.autograd.AutogradValue<V, D, C>`, connecting numerical values,
computation nodes, gradient state, and graph-wide backward entry points.

- Extend `Value<D, C>` and `Accumulatable<V>` with a concrete self-type bound.
- Document node access, stable gradient handles, gradient requirements, and backward seeds.
- Record repeated-pass, derivative-graph, and local-versus-global backward semantics in `SPEC.md`.
- Add no implementation or dependencies; validate the standalone wrapper build.

Status: complete. The interface and specification agree. The standalone
`./mvnw -o verify` build passed with all four existing tests. No implementation
or dependencies were added.

## Completed increment: value-node implementation

Implement `org.ml4j.autograd.node.impl.ValueNodeImpl<V>` against the agreed node contract.

- Store a deferred value supplier and an immutable copy of ordered inputs.
- Provide a leaf constructor and callback installation, replacement, removal, and invocation.
- Test lazy reference access, duplicate inputs, immutable inputs, callbacks, and failure behaviour.
- Use `FloatValue` in tests; add no gradient implementation or dependencies.
- Validate with the standalone Maven Wrapper build.

Status: complete. Added the implementation and seven focused tests. The standalone
`./mvnw -o verify` build passed with all eleven tests. Existing interface contracts
remain unchanged; no dependencies were added.

## Completed increment: node implementation package

Move the value-node implementation and tests from `impl.node` to `node.impl`,
keeping implementation packages alongside their domain interfaces. Update references
and validate the standalone build. Status: complete; `./mvnw -o clean verify` passed
with all eleven tests. Contracts remain unchanged.

## Completed increment: gradient-state implementation

Implement `GradNodeImpl` in `org.ml4j.autograd.node.impl` against the agreed gradient contract.

- Store the first contribution and combine later contributions using non-mutating addition.
- Clear only the held reference on reset; preserve previous expressions and state on addition failure.
- Test absence versus zero, lazy data access, accumulation, reset, nulls, and failed addition
  with a small immutable accumulatable fixture.
- Keep `FloatValue` and all interface contracts unchanged; add no dependencies.
- Validate the standalone Maven Wrapper build.

Status: complete. Added the implementation and seven focused tests. The standalone
`./mvnw -o verify` build passed with all eighteen tests. No interface contracts,
`FloatValue` behaviour, or dependencies changed.

## Completed increment: forward autograd structure

Introduce a forward-only structural demonstration with `AutogradValueImpl`,
`FloatAutogradValue`, and a scalar addition demo.

- Reuse the proven separation between generic graph construction and concrete arithmetic.
- Use tested node implementations; defer self-reference resolution until after construction.
- Implement lazy forward addition, ordered input nodes, and combined gradient requirements.
- Keep backward passes explicitly unimplemented, as requested; document this temporary limitation.
- Test forward behaviour and run the standalone build and demo. Add no dependencies.

Status: complete. Added the base class, float subclass, forward demo, and six tests.
The standalone `./mvnw -o verify` build passed with all twenty-four tests. The demo
prints a sum of 5.0, two inputs, combined gradient requirements, and an absent gradient.
Existing interface contracts are unchanged; backward behaviour remains unimplemented.

## Completed increment: local addition propagation

Implement addition's local backward callback in `AutogradValueImpl`.

- Read the incoming gradient from the result value and accumulate it into each tracked operand.
- Skip absent gradients and constant results/operands; preserve duplicate operand contributions.
- Test explicit seeds, constants, duplicates, and non-mutation using the local node entry point.
- Leave graph-wide backward traversal unimplemented and existing interfaces unchanged.
- Validate the standalone Maven Wrapper build.

Status: complete. Added local propagation and six focused tests. The standalone
`./mvnw -o verify` build passed with all thirty tests. Graph-wide backward entry points
remain unimplemented; interfaces and dependencies are unchanged.

## Completed increment: graph-wide addition backward passes

Implement graph-wide backward execution for addition in `AutogradValueImpl`.

- Traverse inputs in dependency order by node identity; reject cycles before changing state.
- Seed the result explicitly or use the float subclass's default seed of one.
- Propagate only new contributions, replacing intermediate gradients and accumulating leaves.
- Test shared branches, duplicates, constants, repeated passes, resets, and invalid seeds/roots.
- Update the scalar demo and validate the standalone build; leave interface contracts unchanged.

Status: complete. Added backward execution and seven focused tests. The standalone
`./mvnw -o verify` build passed with all thirty-seven tests. The demo shows an absent
constant gradient and a tracked gradient of 1.0. Interface contracts remain unchanged.

## Completed increment: binary derivative contract and callback construction

Introduce `org.ml4j.autograd.operators.BinaryGradient<V>` and generalise local binary callbacks.

- Document explicit incoming-gradient, left-operand, and right-operand parameters in Javadoc and SPEC.md.
- Pass left/right derivative rules into `applyBinaryOperator`, which installs the shared callback.
- Keep addition's two unit-derivative rules and existing propagation behaviour unchanged.
- Test dispatch and distinct operand contributions without adding multiplication.
- Validate the standalone build; add no dependencies or other operations.

Status: complete. Added the interface, matching specification, shared callback construction,
and two dispatch tests. The standalone `./mvnw -o verify` build passed with all
thirty-nine tests. Existing addition behaviour and interface contracts are unchanged.

## Completed increment: scalar multiplication and higher derivatives

Introduce scalar multiplication using the existing generic binary callback construction.

- Add `FloatAutogradValue.mul(other)` with left/right derivative rules and lazy numerical multiplication.
- Record scalar multiplication behaviour in `SPEC.md`; leave existing interface contracts unchanged.
- Test explicit seeds, constants, duplicates, shared branches, repeated passes, and first/second derivatives.
- Extend the scalar demo with the proof-of-concept Hessian-vector example.
- Validate the standalone build and demo; add no dependencies or other numerical operations.

Status: complete. Added multiplication, seven tests, and the Hessian-vector demo.
The standalone `./mvnw -o verify` build passed with all forty-six tests. The demo
prints first derivatives `(1.6, 1.7)` and Hessian-vector product `(5, 4)`.
Existing interface contracts and dependencies remain unchanged.

## Completed increment: unrecorded scalar in-place addition

Historical increment, now reverted by the stable-data decision below. Its mutation
contract and implementation are no longer active.

Introduce unrecorded, shape-preserving scalar `add_()` as the first mutation increment.

- Revise `Value` stability to apply between explicit mutations, preserving functional addition.
- Keep `FloatValue` immutable. Mutable autograd reads expose current contents; its retained supplier is live.
- Add numerical-update machinery to the base class and `FloatAutogradValue.add_()` returning the receiver.
- Preserve context, graph nodes, gradient state, and requirements; self-add uses pre-update contents.
- Document that dependent computations and retained gradient expressions cannot be reused after mutation.
- Test identity, data updates, self-addition, unchanged gradients, and between-pass usage.
- Run the standalone build. Automatic stale-graph detection remains a separate increment.

Status: complete. Contract revision explicitly agreed by the user. Added scalar
`add_()`, generic numerical-update machinery, and six tests. The standalone
`./mvnw -o verify` build passed with all fifty-two tests. `FloatValue` and functional
addition remain unchanged; automatic invalidation checks are not implemented.

## Completed increment: mutation consequences demo

Historical increment, now reverted along with inline operations. The demo was removed.

Add a small scalar mutation demo showing preserved handles, unrecorded updates,
retained lazy dependencies, and the supported between-pass lifecycle.

- Label stale-computation reuse as unsupported and illustrative of missing invalidation checks.
- Include expected-output comments and leave implementation/interface contracts unchanged.
- Validate the standalone build and run the new demo.

Status: complete. The standalone `./mvnw -o verify` build passed with all fifty-two
tests. The demo ran and matched every expected-output comment. No implementation
behaviour or interface contracts changed.

## Completed increment: remove inline operations

Remove inline numerical operations and restore lifetime-stable value data, as explicitly agreed.

- Remove scalar `add_()`, the base numerical-update helper, and their tests/demo.
- Restore `Value` Javadoc and SPEC.md to stable numerical contents and context.
- Keep functional operations, gradient accumulation/reset, and legacy projects unchanged.
- Validate a clean standalone build and the remaining scalar demo.

Status: complete. This supersedes the mutation prototype and its proposed safeguards.
The standalone `./mvnw -o clean verify` build passed with all forty-six remaining
tests, and the first-/second-derivative demo matched its expected output.

## Completed increment: arithmetic wrapper review and cleanup

Review and tidy the user-introduced arithmetic interfaces and float-wrapper implementations.

- Correct copied Javadoc/examples and clarify numerical versus differentiable operations.
- Preserve introduced public signatures while documenting their current responsibilities.
- Fix missing numerical multiplication, incorrect multiplication dispatch, and missing addition
  on the original float implementation/test fixtures after moving addition out of the base.
- Add wrapper arithmetic/gradient tests and validate both demos and the standalone build.
- Record the introduced structure in SPEC.md. Leave legacy code and inline-operation policy unchanged.

Status: complete. Documented the user-introduced interfaces and wrappers, corrected
numerical multiplication/dispatch and primitive-float addition, and added four wrapper tests.
The standalone `./mvnw -o verify` build passed with all fifty tests. Both scalar demos
produced the expected first derivatives and Hessian-vector product. Core contracts are unchanged.

## Completed increment: shared arithmetic defaults

Move wrapped multiplication into the differentiable arithmetic defaults and remove
the redundant protected addData hook from the base class.

- Preserve numerical semantics, raw Float operations, and graph/gradient behaviour.
- Update subclasses and fixtures to provide numerical addition directly when needed.
- Ask explicitly before renaming the public FloatWrapper accessor; preserve it until agreed.
- Update the specification and tests and validate the standalone build and demos.

Status: arithmetic refactoring complete. The standalone `./mvnw -o verify` build passed
with all fifty-one tests, and the wrapped demo matched expected outputs. Multiplication
now uses the interface default; addData was removed. Accessor naming was awaiting
explicit agreement at this stage; the later float-wrapper accessor increment resolves it.

## Current increment: document backward graph policy

Clarify the existing backward behaviour without introducing configuration flags.

- Explain derivative recording (`createGraph`) and graph retention (`keepGraph` or
  `retainGraph`) in SPEC.md and the autograd interface/base-class Javadoc.
- Document higher derivatives, repeated passes, and gradient reset separately.
- Record future discussion triggers: tensor memory pressure, first-order training,
  explicit saved-state release, and operations with limited derivative support.
- Review documentation against the implementation; no runtime changes or new tests.

Status: complete. SPEC.md and interface/base-class Javadoc describe the current
policies and future discussion triggers. Reviewed against backward traversal and
gradient reset; documentation whitespace checks passed. No runtime changes were made.

## Current increment: legacy graph-flag migration note

Document the differing roles of legacy `BackwardConfig.keep_graph` and
`AutogradValueProperties.create_graph` in SPEC.md, including their uneven propagation
and why no equivalent persistent property is carried into the new contract.
Review against legacy source; documentation only.

Status: complete. Added the migration note and checked it against legacy property
assignment, accumulation, callback dispatch, and unary/binary construction.
Documentation whitespace checks passed; no code or contracts changed.

## Current increment: unary negation

- Document numerical negation, its differentiable default, and a public unary integration hook.
- Introduce UnaryGradient with explicit incoming-gradient and forward-operand parameters.
- Implement lazy unary graph construction and negation for raw and wrapped floats.
- Test deferred evaluation, local propagation, constants, composition, repeated passes,
  higher derivatives, and helper argument/context handling. Validate the standalone build.

Status: complete. Added the documented unary contract, shared helper, and raw/wrapped
float negation. Seven focused tests cover the new behaviour, including second derivatives.
The standalone `./mvnw -o verify` build passed with all fifty-eight tests.
No graph flags, dependencies, or legacy changes were introduced.

## Current increment: composed subtraction

- Document numerical sub and differentiable subtraction composed as add(other.neg()).
- Add wrapped numerical subtraction and the default, plus raw Float subtraction.
- Test forward laziness, signed operand gradients, constants, shared operands,
  repeated passes, and second derivatives. Validate the standalone build.

Status: complete. Added numerical subtraction, the composed differentiable default,
raw Float subtraction, and six focused tests. The standalone `./mvnw -o verify`
build passed with all sixty-four tests. No legacy code or dependencies changed.

## Current increment: float-wrapper accessor naming

Rename FloatWrapper.getFloat() to floatValue() following explicit user agreement.
Update implementations, callers, tests, examples, and the active SPEC contract.
Preserve historical decisions as history; validate the standalone build and wrapper demo.

Status: complete. Renamed the interface, implementation, all callers and examples,
and updated SPEC.md. The standalone `./mvnw -o verify` build passed with all
sixty-four tests; the wrapper demo matched its expected outputs. No legacy changes.

## Current increment: arithmetic operation scope

Record all twelve non-inline methods from legacy ArithmeticOperations in SPEC.md,
with implemented/pending status and unresolved semantics distinguished from scope.
Exclude add_ and sub_. This increment documents scope only; review completeness
against the legacy interface without adding signatures or implementation.

Status: complete. SPEC.md lists all twelve methods, identifies the four implemented
methods, and records pending contract decisions. Reviewed against the legacy
interface; no code, public signatures, or established semantics changed.

## Planned arithmetic increments

- [x] Value addition, multiplication, negation, and subtraction.
- [x] Value division, with agreed numerical and derivative rules.
- [ ] Scalar overloads for add, sub, mul, and div, with agreed constant construction.
- [ ] Float-array access, with agreed name and data conversion/ownership policy.
- [ ] Scalar-threshold gt and gte, with agreed result and differentiation policies.

Suggested order only: each increment still requires agreement before implementation.
Inline add_ and sub_ are excluded.

## Current increment: value division

Document and implement div(V) for raw/wrapped floats and the differentiable default.
Use lazy elementwise quotient construction with contributions incoming/right and
-(incoming*left)/(right*right). Float zero and non-finite cases follow IEEE arithmetic.
Test explicit seeds, constants, shared operands, repeated passes, and higher derivatives;
validate the standalone build. Scalar overloads remain a later increment.

Status: complete. Added numerical and differentiable value division for raw/wrapped
floats. Six tests cover lazy evaluation, seeds, constants, shared operands, repeated
passes, higher derivatives, and IEEE boundaries. The standalone `./mvnw -o verify`
build passed with all seventy tests. Scalar overloads remain pending; legacy unchanged.

## Current increment: provisional arithmetic signatures

Expose the seven pending methods for cross-project compatibility review without
implementing numerical operations or derivative rules. Use throwing interface defaults
to preserve existing implementers; DifferentiableArithmeticOperations inherits them.
Document provisional signatures and unresolved semantics in Javadoc and SPEC.md.

Status: complete. Added seven provisional signatures and documented their inherited
unsupported defaults and unresolved contracts. `./mvnw -o verify` passed with all
seventy existing tests and produced the integration JAR. External-project compatibility
has not yet been checked. No numerical or derivative implementation was added.

## Current increment: overridable operator construction

Remove final from the binary and unary graph-construction helpers in AutogradValueImpl.
Document override requirements and delegation for tensor/backend integration.
Preserve signatures and existing helper behaviour.

Status: complete. Both helpers are overridable and their Javadoc and SPEC.md record
the extension requirements. `./mvnw -o verify` passed with all seventy tests.
No helper bodies or tensor adapters changed.

## Current increment: standalone data-access capability

Introduce DataSupplier<D> with deferred Supplier<D> data() access. Make Value<D, C>
extend it while retaining its existing stronger data() contract and declaration.
Document the capability separately from numerical stability and context.

Status: complete. Added the documented interface and Value inheritance; existing
data-access guarantees remain unchanged. `./mvnw -o verify` passed with all seventy
tests. No implementation or dependency changes were needed.

## Current increment: explicit-gradient leaf factory contract

Introduce AutogradValueFactory<V, D, C> with create(data, context, requiresGrad).
Document deferred leaf creation and the explicit gradient choice. Add no concrete
factory, two-argument overload, dependencies, or tensor-project changes.
Record the agreed exclusion of AutogradValueProperties from the new library.

Status: complete. Added the documented factory interface and matching specification.
`./mvnw -o verify` passed with all seventy existing tests. No concrete factory or
tensor-project changes were made.

## Current increment: exclude the core registry

Record the agreed exclusion of AutogradValueRegistry. Keep model parameter lists
at application level; defer tensor registry-call migration and backend resource
ownership/cleanup to separate increments.

Status: complete. SPEC.md records the decision and retained-gradient/resource
limitations. Reviewed against graph traversal and gradient-reset contracts.
Documentation only; no runtime or tensor-project changes, and no build required.

## Current increment: leaf gradient toggling

Add setRequiresGrad(boolean) to AutogradValue and the base implementation. Allow
leaves only, preserve stored gradients/data/nodes, and document graph timing rules.
Test future graph tracking, rejection on operation results, and unchanged state.
Record native-gradient integration as deferred.

Status: complete. Added the setter, leaf-only implementation, Javadoc/specification,
and three focused tests for toggling and unchanged state. `./mvnw -o verify` passed
with all seventy-three tests. Native gradients and tensor-project migration remain deferred.

## Current increment: optional lazy data caching

Add CachingDataSupplier<D> as a final Supplier utility. Cache successful non-null
results, retry failures/nulls, and provide no reset/invalidation. Test deferred reads,
identity, retries, null rejection, and use as an autograd numerical-data supplier.
Keep base-class caching optional and tensor integration deferred.

Status: complete. Added the documented utility, matching specification, and four
focused tests. `./mvnw -o verify` passed with all seventy-seven tests. Core caching
policy and tensor-project code remain unchanged.

## Current increment: stable parameter-handle contract

Introduce Parameter<V> with value() and replace(V), separating replaceable parameter
selection from stable Value numerical data. Document leaves, null/error handling,
reference/gradient preservation, model compatibility, and existing graph semantics.
Add only the interface and update Value documentation; defer implementation and
model migration.

Status: complete. Added Parameter and linked the responsibility distinction in
Value Javadoc and SPEC.md. `./mvnw -o verify` passed with all seventy-seven tests.
No concrete parameter implementation or tensor-project changes were added.

## Current increment: parameter-handle implementation

Add ParameterImpl in the core impl package, validating non-null leaves before
assignment. Test lazy access/replacement, preserved gradient/flag/node state,
invalid-input handling, and old/new graph differentiation after replacement.
Leave tensor compatibility to callers and defer model integration.

Status: complete. Added the documented implementation and three focused tests.
`./mvnw -o verify` passed with all eighty tests. No tensor-project or dependency
changes were made.

## Current increment: gradient-access migration decision

Keep Optional gradient access and explicit gradient-node reset. Add no legacy aliases.

Status: complete. Recorded caller migration and stable-handle/replaced-expression
semantics in SPEC.md; reviewed against existing GradNode and AutogradValue contracts.
Documentation only; no build or tensor-project changes.

## Current increment: tensor implicit-seed migration decision

Use a constant same-shape leaf of ones from tensor defaultGradient() overrides.
Preserve explicit backward seeds and the generic core's existing default behaviour.

Status: complete. Recorded the policy in SPEC.md and checked it against the existing
backward/defaultGradient contract. Documentation only; no tensor changes or build.

## Current increment: backward-configuration migration decision

Exclude BackwardConfig and migrate callers to the existing unconfigured backward
methods. Keep native gradients and configurable graph policies deferred.

Status: complete. SPEC.md records the decision, reviewed against existing backward
policies. Documentation only; no runtime or tensor-project changes and no build.

## Current increment: broadcast-gradient migration decision

Retain tensor broadcast-gradient reduction through the overridable binary helper,
using the new BinaryGradient signature and delegation to the base implementation.

Status: complete. Recorded adapter responsibilities and migration validation scope
in SPEC.md, consistent with the operator extension contract. Documentation only;
no tensor changes or build.

## Current increment: tensor scalar-overload migration decision

Preserve legacy unary graph construction for fixed float constants, retaining tensor
shape and differentiable contribution rules without constructing scalar leaves.

Status: complete. Recorded forward/derivative rules and migration scope in SPEC.md.
Documentation only; provisional defaults and tensor code are unchanged. No build.

## Current increment: tensor comparison-mask migration decision

Use lazy constant leaves for gt/gte masks, with no graph inputs or backward callback.
Retain mask use in ReLU's derivative and defer boundary validation to implementation.

Status: complete. Recorded construction, differentiation, and validation requirements
in SPEC.md. Documentation only; no runtime/tensor changes or build.

## Current increment: float-array ownership decision

Allow dataAsFloatArray to return copied or shared storage with read-only caller
usage. Backend flattening/conversion is documented during migration; editable access
requires an explicit caller copy.

Status: complete. Recorded ownership and access semantics in SPEC.md. Documentation
only; provisional defaults and tensor implementations are unchanged. No build.

## Current increment: tensor inline-operation removal decision

Remove adapter add_/sub_ methods during migration and use stable-value arithmetic
and parameter replacement. Check production/test callers before removal.

Status: complete. Source search found only adapter declarations and DL4J mutation-test
calls; recorded their migration in SPEC.md. No runtime/tensor changes or build.

## Current increment: parameter naming decision

Keep names in model maps associated with stable Parameter handles; restore neither
name() nor name_() in core interfaces. Defer optional operation/debug metadata.

Status: complete. Recorded the decision in SPEC.md. Documentation only; no runtime
or tensor-project changes and no build.

## Current increment: defer tensor copying

Exclude tensor copy support for now; leave storage and graph semantics unresolved.
Review existing copy-constructor callers during migration before removing unsupported
paths, without adding core copy constructors.

Status: complete. Recorded scope in SPEC.md. Documentation only; no runtime/tensor
changes or build.

## Current increment: share scalar and comparison defaults

Move six tensor overrides into DifferentiableArithmeticOperations, add an overridable
constant-leaf helper for masks, and remove redundant tensor overrides. Verify lazy
forward evaluation, seeds/higher derivatives, mask state, and both backend suites.
Numerical scalar implementations and float-array delegation are outside this move.

Status: complete. Core defaults and constant-leaf helper added; redundant tensor
overrides removed. Core verify passed with 84 tests, artifact install passed, and
tensor clean verify passed with 81 tests across both backends and GPT smoke tests.

## Current increment: declarations-only ArithmeticOperations

Remove throwing defaults and update Javadoc. Supply required concrete float numerical
methods and wrapped array access; use production numerical operations in shared tests.
Validate numerical boundaries and rebuild core/tensor projects.

Status: complete. ArithmeticOperations has no implementations; concrete float
operations and wrapped array access satisfy abstract signatures. Core verify passed
with 87 tests; artifact installed; tensor clean verify passed with 81 tests.

## Current increment: activation cleanup and ReLU

Document numerical/differentiable activation contracts, supply shared unary ReLU
and float numerical implementation, and tidy demo numerical output. Test boundaries,
laziness, seeds/constants, repeated passes, higher derivatives, and demo regression.

Status: complete. Documented activation contracts and generic capability bounds,
implemented numerical/shared ReLU, and tidied demo output. Six focused tests cover
boundaries, lazy construction, seeds/constants, shared/repeated passes, higher
derivatives, and the micrograd regression. `./mvnw -o verify` passed with 93 tests.
AutogradDemo ran and printed 24.704082, 138.83382, and 645.5772.

## Completed increment: mathematical functions and tensor migration

- [x] Introduce org.ml4j.autograd.math.MathOperations and DifferentiableMathOperations.
- [x] Document a behavior-preserving migration: existing derivative formulas and
  backend domains remain; no zero/one power special cases or common boundary policy.
- [x] Supply numerical float exp/log/pow and shared lazy differentiable defaults.
- [x] Adopt capabilities in scalar and tensor types; remove tensor-local declarations
  and derivative overrides while retaining DJL/DL4J numerical implementations.
- [x] Test boundaries, laziness, graph state, constants, seeded/shared/repeated passes,
  and second derivatives in core; test tensor shape, caching, operand preservation,
  seeded/repeated/second derivatives, including pow(-0.5), on both backends.
- [x] Verify core, install artifact, and validate both tensor backends and GPT.

Verification: core `./mvnw -o -B clean verify` passed with 102 tests; artifact
installation used `./mvnw -o -B -DskipTests install`. Tensor
`./mvnw -o -B clean verify` passed with 85 tests, including deterministic GPT
training/inference on both backends (native-cache access enabled).

Power boundary improvements and uniform backend domain semantics remain separate
open decisions. The general rule may yield NaN at zero for pow(0) and at higher
polynomial derivative orders. Softmax/reductions remain in the tensor project.


## Completed increment: shared sigmoid

- [x] Add sigmoid to numerical/shared differentiable activation capabilities.
- [x] Supply stable float numerical evaluation and the shared incoming * s * (1 - s) rule.
- [x] Remove tensor-local declaration and derivative override; retain DJL/DL4J numerics.
- [x] Test scalar boundaries, lazy construction, constants, seeded/repeated passes,
  and second derivatives; test tensor shape, caching, preservation, repeated and
  higher derivatives on both backends.

Verification: autograd `./mvnw -o -B verify` passed with 98 tests; the updated
artifact was installed with `./mvnw -o -B -DskipTests install`. Tensor
`./mvnw -o -B clean verify` passed with 83 tests, including deterministic GPT
training/inference on both backends. Initial sandboxed tensor execution could not
extract native libraries; rerunning with native-cache access passed.


## Scalar package relocation

Move wrapped scalar interfaces, implementations, tests and demo from
org.ml4j.autograd.arithmetic.scalar to org.ml4j.autograd.scalar. Update imports
and documentation to reflect their arithmetic, activation and math capabilities.
Status: complete. Core `./mvnw -o -B clean verify` passed with 102 tests.
Installed the updated artifact with `./mvnw -o -B -DskipTests install`.
Downstream tensor `./mvnw -o -B clean verify` passed with 85 tests,
including both backend suites and GPT training/inference (native-cache access enabled).


## Scalar demo relocation

Move FloatAutogradValueDemo and FloatWrapperAutogradValueDemo into
org.ml4j.autograd.demo alongside AutogradDemo, and update documented entry points.
Status: complete. `./mvnw -o -B clean verify` passed with 102 tests. Both
relocated entry points ran with `java -cp target/classes org.ml4j.autograd.demo.<DemoClass>`
and printed the expected forward values, gradients and Hessian-vector products.


## Cache utility relocation

Move CachingDataSupplier and its tests into org.ml4j.autograd.cache and update
the tensor adapter import and documentation.
Status: complete. Core and tensor `./mvnw -o -B clean verify` passed with
102 and 85 tests respectively, including both backend suites and GPT training/inference.
Installed the updated core artifact with `./mvnw -o -B -DskipTests install`;
tensor verification used native-cache access.


## Resume point and remaining decisions (2026-10-14)

Current stopping point: core graph traversal, gradient accumulation/reset, repeated
passes, higher derivatives and stable Parameter handles are implemented. Shared
arithmetic/comparison defaults, ReLU/sigmoid, and exp/log/pow are adopted by the
tensor project; DJL/DL4J numerics, shape/broadcast handling, matrix operations,
reductions and softmax remain there. Scalar implementations live in scalar,
CachingDataSupplier in cache, and runnable scalar examples in demo.

Last verified state: core and tensor `./mvnw -o -B clean verify` passed with
102 and 85 tests respectively, including both backend suites and deterministic GPT
training/inference. The updated autograd 3.0.0-SNAPSHOT artifact is installed locally.
Native tensor tests required cache access. These are historical results, not new
verification of future changes. Both relocated scalar demos were also run successfully.

The detailed questions and existing limitations are recorded in SPEC.md under
"Open decisions for continuing autograd work (2026-10-14)". No next implementation
has been selected or authorized:

- [ ] Agree shared reduction/context contracts and then softmax placement/dependencies.
- [ ] Agree power zero/one special cases, higher-order semantics and boundary policies.
- [ ] Decide whether/how to integrate native gradients while preserving core contracts.
- [ ] Agree tensor storage copying versus graph retention/detachment semantics.
- [ ] Agree and validate native resource ownership for sustained training.

When resuming, choose one topic, review its current implementation and open questions,
agree the smallest contract increment, then implement/test and update SPEC.md and
PLAN.md. For shared core changes, verify/install core before downstream verification.
Use both backend suites and deterministic GPT tests for tensor-facing changes.
Native ownership work additionally needs sustained memory/lifetime validation.
These topics are independent discussion tracks; this list does not prescribe an
implementation order or authorize contract changes. Read ../minimal-maven-project/SPEC.md
and PLAN.md when work crosses into the tensor project.


## Wrapped scalar factory pattern

Add the typed FloatWrapperAutogradValueFactory interface and delegating scalar
implementation, with primitive/explicit-tracking and inherited supplier creation.
Verify fresh leaf state, generic factory use, lazy/non-cached suppliers, null supplier
rejection and derivative integration. Existing constructor semantics remain unchanged.
Status: complete. `./mvnw -o -B clean verify` passed all 105 tests, including
three factory tests for leaf independence/tracking, generic lazy supplier creation,
null supplier rejection and repeated/higher-order backward integration.


## Separate wrapped scalar numerical and autograd tests

Keep the numerical arithmetic test in FloatWrapperTest. Move the four existing
tracking/gradient, higher-derivative and repeated-pass tests unchanged into
FloatWrapperAutogradValueTest. `./mvnw -o -B clean verify` passed all 105 tests; the test count is unchanged.


## Factory-backed wrapped scalar behavior fixtures

Migrated wrapped-value creation in nine further behavioral test classes to the
FloatWrapperAutogradValueFactory interface. Existing numerical objects and deferred
suppliers use inherited create(supplier, null, requiresGrad); primitive fixtures use
the scalar overload. Raw FloatAutogradValue fixtures and assertions are unchanged.
Verification: ./mvnw -o -B clean verify passed all 105 tests.


## Wrapped scalar implementation package

Move the three wrapped scalar *Impl classes and four focused tests into scalar.impl.
Update imports, demos and links; retain interfaces/raw scalar types and shared
operation tests in their current packages. Behavior unchanged.
Verification: ./mvnw -o -B clean verify passed all 105 tests.


## Raw scalar interface and factory pattern

Split FloatAutogradValue into scalar interface and scalar.impl implementation;
add typed factory and delegating implementation. Retain arithmetic/graph behavior,
update constructor callers/demos and move constructor-focused tests to scalar.impl.
Clarify raw Float versus wrapped numerical capability Javadoc. Verification: ./mvnw -o -B clean verify passed all 108 tests, including three
new raw factory tests. The factory-backed raw demo printed expected forward,
gradient and Hessian-vector results. Scalar Javadoc generation passed doclint
syntax/link checks (missing-comment warnings excluded).


## Factory-backed raw scalar behavior fixtures

Migrated raw scalar fixtures in nine behavioral suites, including BackwardPassTest,
parameter lifecycle and caching integration, to the FloatAutogradValueFactory
interface. Deferred/cached suppliers use create(supplier, null, requiresGrad).
Kept direct construction in the constructor-focused FloatAutogradValueTest and
for the two implementation-only unary-helper receivers in NegationTest. Seeds and
contributions in those helper tests use the factory. Assertions remain unchanged.
Verification: ./mvnw -o -B clean verify passed all 108 tests.


## Factory-backed AutogradDemo

Use an interface-typed FloatWrapperAutogradValueFactory in the micrograd demo
for tracked inputs and constants. Remove the constructor helper; preserve arithmetic
and printed forward/gradient values.
Verification: `./mvnw -o -B clean verify` passed all 108 tests. Running
AutogradDemo printed the expected 24.704082, 138.83382 and 645.5772.


## Extensible and supplied gradient nodes

- [x] Make GradNodeImpl non-final. Document equivalent lazy read views and the
  requirement to preserve absence/zero, numerical stability and derivative graphs.
- [x] Add a protected AutogradValueImpl constructor accepting a non-null GradNode
  after the existing inputs argument. Preserve the original constructor/default
  accumulator and final stable gradNode() accessor; do not call subclass/node hooks
  or evaluate numerical data during construction.
- [x] Verify supplied-node identity, independent defaults, null rejection, inherited
  raw accumulation/reset, lazy view reads, duplicate/shared contributions, frozen
  inputs, repeated passes, retained values and higher derivatives.

Verification: `./mvnw -B -o clean verify` passed all 114 tests (six new extension
checks), zero failures/errors/skips on JDK 25. Whitespace checks passed. No new
interfaces, dependencies, native-gradient suppliers or backend concepts introduced.
Changes are limited to autograd; no artifacts installed or commits made. The
corresponding tensor-base constructor and DJL gradient-node integration are later
increments, not implemented here.


## Successful backward completion hook

- [x] Add protected no-op afterBackward(List<ValueNode<?>> visitedNodes) on the root.
- [x] Invoke after propagation and previous-leaf-gradient restoration, using an
  unmodifiable copy of the existing traversal; no second graph walk or numerical reads.
- [x] Test final restored totals, duplicate-node identity, constants, implicit/explicit
  seeds, repeated calls, lazy data, failed propagation and hook exception behavior.

Verification: ./mvnw -B -o clean verify passed 118 tests, zero failures/errors/skips.
No artifacts installed, commits made or DJL integration changes in this increment.


## Extract autograd operations capability

Introduce AutogradOperations<V> with requiresGrad(), setRequiresGrad(boolean enabled),
backward(), and backward(V gradient). AutogradValue extends it and inherits the four
methods. Preserve all existing tracking and backward contracts and move their Javadoc
with the declarations. No implementation or numerical-mutation behaviour changes.

Status: complete. SPEC.md records the capability contract. A focused test exercises
tracking, both backward overloads, accumulation and invalid calls through the new
interface. `./mvnw verify` passed all 119 tests with no failures, errors or skips.
Changes remain within this repository; no downstream build or artifact installation.


## Receiver-local gradient reset capability

Add resetGradient() to AutogradOperations and a default AutogradValue implementation
that delegates to gradNode().reset(). Reset clears only the receiver's gradient to
absence, is idempotent and lazy, preserves retained expressions and graph state,
and does not release graph data or native resources. This supersedes the earlier
node-only reset API decision; tensor wrappers should delegate to their wrapped value.

Status: complete. Updated specification and Javadoc. Two focused tests cover local
reset on leaves/results, absent gradients, unchanged tracking/node identity, deferred
data, graph reuse and retained higher derivatives. `./mvnw verify` passed all 121
tests with no failures, errors or skips. Tensor adapter implementation and downstream
verification remain outside this repository increment; no artifact installation.


## Generic parameter handles with autograd extraction

Remove the Parameter<V> bound so clients can hold tensor wrappers. ParameterImpl<V>
accepts an extraction function for leaf validation; retain the direct-value constructor
with runtime AutogradValue checks. Preserve lazy access, leaf validation and atomic
reference replacement on failure. Document extractor stability, nulls and exceptions.
Status: complete. ./mvnw verify passed all 124 tests with no failures, errors
or skips. Focused tests cover wrapper graph retention, extraction failures, deferred
data and direct-constructor validation. No downstream project verification performed.


## Gradient read capability

Add Optional<V> gradient() to AutogradOperations and default delegation through
AutogradValue.gradNode().gradient(). Preserve absent/zero distinctions, lazy reads,
compatible gradients and retained derivative expressions; V remains unbounded for
wrapper gradient types. Direct capability implementations must supply the accessor.
Status: complete. ./mvnw verify passed all 126 tests with no failures, errors or
skips. New tests cover absent/zero gradients, deferred reads, accumulation, reset
and retained higher derivatives through the capability. No tensor adapter implementation
or downstream verification in this increment.


## Operation construction diagnostics

Centralize construction-origin capture and exception annotation in private helpers
inside AutogradValueImpl. Share each operation's origin between forward evaluation
and the entire local backward callback; preserve lazy/non-cached data and original
RuntimeException identity. Attach only the first origin marker. Leaves and constant
operations are covered; diagnostic configuration remains deferred.

Status: complete. ./mvnw verify passed all 130 tests, including four new diagnostic
tests covering nested/repeated failures, null forward results, unary derivative
failures, binary accumulation failures and lazy gradient data. No dependencies or
public interfaces added. No downstream build or artifact installation performed.

## Phase labels for operation diagnostics

Added phase labels for forward evaluation, unary/left/right gradient calculation
and accumulation, plus the remaining backward callback. Each failure diagnostic
reuses the operation construction stack; no extra construction stacks are captured.
Verification after removal of the temporary injected failure: ./mvnw verify passed
all 131 tests, including five diagnostic tests, with no failures, errors or skips.
