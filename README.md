# autograd

A standalone Java automatic differentiation project. This repository currently
contains the Maven setup; interfaces, implementations, and tests will be added incrementally.

Requires JDK 25. The Maven Wrapper uses Maven 3.9.12.

Build from the repository root:

```sh
./mvnw verify
```

On Windows, use `mvnw.cmd verify`. The first run downloads Maven if needed.

Licensed under the Apache License 2.0; see [LICENSE](LICENSE).
