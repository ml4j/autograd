package org.ml4j.autograd.demo;

import org.ml4j.autograd.scalar.FloatWrapperAutogradValue;
import org.ml4j.autograd.scalar.impl.FloatWrapperAutogradValueImpl;

/**
 * Runs the scalar arithmetic/ReLU example from Andrej Karpathy's micrograd demo.
 * Prints the forward result and gradients of its two inputs using library rules.
 */
public final class AutogradDemo {
    private AutogradDemo() { }


    /** Runs the example; command-line arguments are unused. */
    public static void main(String[] args) {

        // a = Value(-4.0)
        var a = createAutogradValue(-4.0f, true);

        // b = Value(2.0)
        var b = createAutogradValue(2.0f, true);

        // c = a + b
        var c = a.add(b);

        // d = a * b + b**3
        var d = a.mul(b).add(b.mul(b).mul(b));

        // c += c + 1
        c = c.add(c.add(1));

        // c += 1 + c + (-a)
        c = c.add(createAutogradValue(1.0f, false).add(c).add(a.neg()));

        // d += d * 2 + (b + a).relu()
        d = d.add(d.mul(2).add(b.add(a).relu()));

        // d += 3 * d + (b - a).relu()
        d = d.add(createAutogradValue(3.0f, false).mul(d).add(b.sub(a).relu()));

        // e = c - d
        var e = c.sub(d);

        // f = e**2
        var f = e.mul(e);

        // g = f / 2.0
        var g = f.div(2.0f);

        // g += 10.0 / f
        g = g.add(createAutogradValue(10.0f, false).div(f));

        System.out.println(g.data().get().floatValue()); // Prints 24.704082, the outcome of this forward pass

        // Perform a backward pass
        g.backward();

        System.out.println(a.gradNode().gradient().orElseThrow().data().get().floatValue()); // Prints 138.83382, i.e. the numerical value of dg/da

        System.out.println(b.gradNode().gradient().orElseThrow().data().get().floatValue()); // Prints 645.5772, i.e. the numerical value of dg/db

    }

    private static FloatWrapperAutogradValue createAutogradValue(float value, boolean requiresGrad) {
        return new FloatWrapperAutogradValueImpl(value, requiresGrad);
    }
}
