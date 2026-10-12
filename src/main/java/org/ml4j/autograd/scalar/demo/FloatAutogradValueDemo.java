package org.ml4j.autograd.scalar.demo;

import org.ml4j.autograd.scalar.FloatAutogradValue;

/** Demonstrates scalar addition, graph structure, and a graph-wide backward pass. */
public final class FloatAutogradValueDemo {
    private FloatAutogradValueDemo() {}

    /**
     * Runs the scalar addition example with a constant and a tracked operand.
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
        sum.backward();
        System.out.println("Constant gradient: " + first.gradNode().gradient());
        System.out.println("Tracked gradient: " + second.gradNode().gradient().orElseThrow().data().get());
    }
}
