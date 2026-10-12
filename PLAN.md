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
- [ ] Validate repeated passes, shared inputs, resets, and higher derivatives.

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

## Next increment

Introduce scalar multiplication and validate first derivatives and Hessian-vector
products when requested. No further operations are authorised by this plan alone.
