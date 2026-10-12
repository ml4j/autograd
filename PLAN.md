# Development plan

## Approach

For each increment, define a small interface and its purpose, review its contract,
then add an implementation and tests when requested. Validate the increment before
moving to the next. This plan records direction, not permission to implement all stages.

## Progress

- [x] Standalone Maven setup and wrapper.
- [x] Initial agent instructions, specification, and plan.
- [ ] Introduce the numerical value/data-access contract.
- [ ] Implement and test basic data access.
- [ ] Introduce and test gradient combination and state contracts.
- [ ] Introduce computation-node and local propagation contracts.
- [ ] Implement scalar operations and backward traversal.
- [ ] Validate repeated passes, shared inputs, resets, and higher derivatives.

## Next increment

Review the minimal value interface: numerical data access, laziness, context, and
data stability. Agree its contract before adding an implementation.

