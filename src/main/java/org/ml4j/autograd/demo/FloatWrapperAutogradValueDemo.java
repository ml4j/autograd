package org.ml4j.autograd.demo;

import org.ml4j.autograd.scalar.impl.FloatWrapperAutogradValueImpl;

/** Demonstrates shared arithmetic defaults with wrapped scalar data and a Hessian-vector product. */
public final class FloatWrapperAutogradValueDemo {
    private FloatWrapperAutogradValueDemo() {}

    /**
     * Runs first- and second-derivative examples, unwrapping numerical data with floatValue().
     * @param args unused command-line arguments
     */
    public static void main(String[] args) {
        var first = new FloatWrapperAutogradValueImpl(2f, false);
        var second = new FloatWrapperAutogradValueImpl(3f, true);
        var sum = first.add(second);
        System.out.println("Forward sum: " + sum.data().get().floatValue()); // Forward sum: 5.0
        System.out.println("Input count: " + sum.valueNode().inputs().size()); // Input count: 2
        System.out.println("Node represents sum: " + (sum.valueNode().value() == sum)); // Node represents sum: true
        System.out.println("Requires gradient: " + sum.requiresGrad()); // Requires gradient: true
        System.out.println("Gradient before backward: " + sum.gradNode().gradient()); // Gradient before backward: Optional.empty
        sum.backward();
        System.out.println("Constant gradient: " + first.gradNode().gradient()); // Constant gradient: Optional.empty
        System.out.println("Tracked gradient: " + second.gradNode().gradient().orElseThrow().data().get().floatValue()); // Tracked gradient: 1.0

        var x = new FloatWrapperAutogradValueImpl(0.5f, true);
        var y = new FloatWrapperAutogradValueImpl(0.6f, true);
        var z = x.mul(x).add(y.mul(x)).add(y.mul(y));
        z.backward();
        var gx = x.gradNode().gradient().orElseThrow();
        var gy = y.gradNode().gradient().orElseThrow();
        System.out.println("First derivatives: " + gx.data().get().floatValue() + ", " + gy.data().get().floatValue()); // First derivatives: 1.6, 1.7
        // Reset the state handles, retaining the gradient expressions and their own graphs.
        x.gradNode().reset();
        y.gradNode().reset();
        gx.mul(new FloatWrapperAutogradValueImpl(2f, false)).add(gy).backward();
        System.out.println("Hessian times (2, 1): " + x.gradNode().gradient().orElseThrow().data().get().floatValue()
                + ", " + y.gradNode().gradient().orElseThrow().data().get().floatValue()); // Hessian times (2, 1): 5.0, 4.0
    }
}
