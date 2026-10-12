package org.ml4j.autograd.scalar.demo;

import org.ml4j.autograd.scalar.FloatAutogradValue;

/** Demonstrates scalar forward addition and graph structure; backward execution is intentionally deferred. */
public final class FloatAutogradValueDemo {
    private FloatAutogradValueDemo() {}

    /**
     * Runs the forward-only example.
     * @param args unused command-line arguments
     */
    public static void main(String[] args) {
        var first = new FloatAutogradValue(2f, false);
        var second = new FloatAutogradValue(3f, true);
        var sum = first.add(second);
        System.out.println("Forward sum: " + sum.data().get());
        System.out.println("Input count: " + sum.valueNode().inputs().size());
        System.out.println("Node represents sum: " + (sum.valueNode().value() == sum));
        System.out.println("Requires gradient: " + sum.requiresGrad());
        System.out.println("Gradient before backward: " + sum.gradNode().gradient());
        System.out.println("Backward execution is not implemented yet.");
    }
}
