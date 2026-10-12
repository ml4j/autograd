# Development plan

## Approach

For each increment, define a small interface and its purpose, review its contract,
then add an implementation and tests when requested. Validate the increment before
moving to the next. This plan records direction, not permission to implement all stages.

## Progress

- [x] Standalone Maven setup and wrapper.
- [x] Initial agent instructions, specification, and plan.
- [x] Introduce the numerical value/data-access contract.
- [ ] Implement and test basic data access.
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

## Next increment

Implement and test basic data access against the agreed `Value` contract when requested.
