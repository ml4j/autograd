# Agent instructions

This is a standalone Java 25 Maven library. It must build without a parent project
or sibling repositories. Work within this repository; other projects are optional
reference material, not dependencies.

- Read `SPEC.md` and `PLAN.md` before making changes.
- Implement only the increment requested by the user. Introduce documented contracts
  before their implementations; keep changes small and reviewable.
- Record agreed interface decisions in `SPEC.md` when introducing each interface.
  Do not change an established interface contract without the user's explicit agreement.
  Propose changes and their consequences first; implementation convenience or a later
  plan stage does not authorise a contract change.
- Use concise Javadoc to explain each public type's purpose and observable behaviour.
  Name read accessors `x()` and actions with verbs.
- Add meaningful tests with implementation. Add dependencies only when the current
  increment needs them.
- Validate code and build changes with `./mvnw verify` (`mvnw.cmd verify` on Windows).
  Report what was checked and any limitations. Documentation-only changes need a
  consistency review rather than a build.
- Keep the specification and plan aligned with agreed changes. Mark unresolved
  decisions explicitly; do not treat future plan stages as authorisation to implement them.
