package org.ml4j.autograd.demo;

import org.ml4j.autograd.AutogradValueRegistry;
import org.ml4j.autograd.demo.scalar.DemoFloatAutogradValueImpl;
import org.ml4j.autograd.impl.AutogradValueProperties;

/**
 * @author Michael Lavelle
 *
 * Inspired by the example usage at https://github.com/karpathy/micrograd?tab=readme-ov-file#example-usage
 */
public class AutogradDemo {

    static AutogradValueRegistry registry;
    static DemoSize size = new DemoSize() {
    };

    static {
        registry = AutogradValueRegistry.create(AutogradDemo.class.getName());

    }
    public static void main(String[] args) {

        // a = Value(-4.0)
        var a = createGradValue(-4.0f, true);

        // b = Value(2.0)
        var b = createGradValue(2.0f, true);

        // c = a + b
        var c = a.add(b);

        // d = a * b + b**3
        var d = a.mul(b).add(b.mul(b).mul(b));

        // c += c + 1
        c = c.add(c.add(1));

        // c += 1 + c + (-a)
        c = c.add(createGradValue(1.0f, false).add(c).add(a.neg()));

        // d += d * 2 + (b + a).relu()
        d = d.add(d.mul(2).add(b.add(a).relu()));

        // d += 3 * d + (b - a).relu()
        d = d.add(createGradValue(3.0f, false).mul(d).add(b.sub(a).relu()));

        // e = c - d
        var e = c.sub(d);

        // f = e**2
        var f = e.mul(e);

        // g = f / 2.0
        var g = f.div(2.0f);

        // g += 10.0 / f
        g = g.add(createGradValue(10.0f, false).div(f));

        System.out.println(g.data().get()); // Prints 24.704082, the outcome of this forward pass

        // Perform a backward pass
        g.backward();
        
        System.out.println(a.grad().data().get()); // Prints 138.83382, i.e. the numerical value of dg/da

        System.out.println(b.grad().data().get()); // Prints 645.5772, i.e. the numerical value of dg/db

    }

    protected static DemoAutogradValue<Float> createGradValue(Float value, boolean requires_grad) {
        return new DemoFloatAutogradValueImpl(new AutogradValueProperties<DemoSize>().setContext(size).setRegistry(registry).setRequires_grad(requires_grad), () -> value);
    }
}
