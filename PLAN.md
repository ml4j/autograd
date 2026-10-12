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
- [ ] Introduce and test gradient combination and state contracts.
- [ ] Introduce computation-node and local propagation contracts.
- [ ] Implement scalar operations and backward traversal.
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

## Next increment

Review the gradient-combination contract before introducing another interface,
when requested. No subsequent increment is authorised by this plan alone.
