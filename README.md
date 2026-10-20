# autograd

A generic automatic differentiation library written in Java. It implements
backpropagation (reverse-mode autodiff) over a dynamically built computation graph
for numerical structures such as scalars, vectors, matrices or tensors, depending
on the implementation.

Inspired by Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd).
Implementations supply numerical operations and context such as tensor shape;
the core is independent of numerical storage and backend libraries. This repository
includes scalar implementations and examples.

## Example usage

This example from [AutogradDemo](src/main/java/org/ml4j/autograd/demo/AutogradDemo.java)
combines scalar arithmetic and ReLU, then calculates the gradients of two inputs.
The factory creates leaves with explicit gradient tracking: `true` for inputs to
differentiate and `false` for constants.

```java
import org.ml4j.autograd.scalar.FloatWrapperAutogradValueFactory;
import org.ml4j.autograd.scalar.impl.FloatWrapperAutogradValueFactoryImpl;

FloatWrapperAutogradValueFactory factory = new FloatWrapperAutogradValueFactoryImpl();

// a = Value(-4.0)
var a = factory.create(-4.0f, true);

// b = Value(2.0)
var b = factory.create(2.0f, true);

// c = a + b
var c = a.add(b);

// d = a * b + b**3
var d = a.mul(b).add(b.mul(b).mul(b));

// c += c + 1
c = c.add(c.add(1));

// c += 1 + c + (-a)
c = c.add(factory.create(1.0f, false).add(c).add(a.neg()));

// d += d * 2 + (b + a).relu()
d = d.add(d.mul(2).add(b.add(a).relu()));

// d += 3 * d + (b - a).relu()
d = d.add(factory.create(3.0f, false).mul(d).add(b.sub(a).relu()));

// e = c - d
var e = c.sub(d);

// f = e**2
var f = e.mul(e);

// g = f / 2.0
var g = f.div(2.0f);

// g += 10.0 / f
g = g.add(factory.create(10.0f, false).div(f));

System.out.println(g.data().get().floatValue()); // Prints 24.704082, the outcome of this forward pass

// Perform a backward pass
g.backward();

System.out.println(a.gradient().orElseThrow().data().get().floatValue()); // Prints 138.83382, i.e. the numerical value of dg/da

System.out.println(b.gradient().orElseThrow().data().get().floatValue()); // Prints 645.5772, i.e. the numerical value of dg/db
```

The forward result is `24.704082`; the gradients are `138.83382` for `a` and
`645.5772` for `b`. `gradNode().gradient()` returns an Optional because a gradient
may not yet have been calculated. Numerical data is read through `data().get()`.

Backward retains differentiable gradient expressions for higher derivatives.
Repeated passes accumulate leaf gradients; call `gradNode().reset()` to clear them.

## Build and run

Requires JDK 25. The Maven Wrapper pins Maven 3.9.12 and downloads it on first use
if needed. Build and run the example from the repository root:

```sh
./mvnw verify
java -cp target/classes org.ml4j.autograd.demo.AutogradDemo
```

On Windows, use `mvnw.cmd verify` for the build.

See [SPEC.md](SPEC.md) for contracts, [PLAN.md](PLAN.md) for development decisions,
and [AGENTS.md](AGENTS.md) for contributor guidance.

## License

Licensed under the Apache License 2.0; see [LICENSE](LICENSE).
