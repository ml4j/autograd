package org.ml4j.autograd.scalar;

import org.ml4j.autograd.impl.AutogradValueImpl;
import org.ml4j.autograd.node.ValueNode;

import java.util.List;
import java.util.function.Supplier;

/**
 * Demonstrates a scalar implementation of the shared autograd structure.
 *
 * <pre>{@code
 * var first = new FloatAutogradValue(2f, true);
 * var second = new FloatAutogradValue(3f, false);
 * var sum = first.add(second);
 * Float data = sum.data().get(); // 5.0
 * }</pre>
 *
 * <p>Uses Void context because scalars need no shape metadata. Unlike a tensor
 * subclass, numerical addition is ordinary float addition. Forward results are lazy;
 * supplied numerical data must remain stable. Addition supports local and graph-wide
 * propagation; the implicit scalar seed is one.
 */
public final class FloatAutogradValue extends AutogradValueImpl<FloatAutogradValue, Float, Void> {

    /**
     * Creates a scalar leaf from fixed numerical data.
     * @param value the scalar, including any valid primitive float value
     * @param requiresGrad whether a future backward pass should calculate its gradient
     */
    public FloatAutogradValue(float value, boolean requiresGrad) {
        this(() -> value, requiresGrad);
    }

    /**
     * Creates a scalar leaf with deferred numerical data.
     * @param data a supplier of stable, non-null float data
     * @param requiresGrad whether a future backward pass should calculate its gradient
     * @throws NullPointerException if the supplier is null
     */
    public FloatAutogradValue(Supplier<Float> data, boolean requiresGrad) {
        this(data, requiresGrad, List.of());
    }

    private FloatAutogradValue(Supplier<Float> data, boolean requiresGrad, List<? extends ValueNode<?>> inputs) {
        super(data, null, requiresGrad, inputs);
    }

    /** Returns this scalar as the node's concrete owner. */
    @Override protected FloatAutogradValue self() { return this; }

    /** Supplies a constant scalar seed of one for implicit backward execution. */
    @Override protected FloatAutogradValue defaultGradient() {
        return new FloatAutogradValue(1f, false);
    }

    /** Adds scalar numerical data without changing either operand. */
    @Override protected Float addData(Float left, Float right) { return left + right; }

    /** Scalars need no context metadata. */
    @Override protected Void mapContext(Void left, Void right) { return null; }

    /** Constructs the lazy scalar result with its ordered operands and combined gradient requirement. */
    @Override
    protected FloatAutogradValue createAutogradValue(Supplier<Float> dataSupplier, Void context,
            boolean requiresGrad, List<? extends ValueNode<?>> inputs) {
        return new FloatAutogradValue(dataSupplier, requiresGrad, inputs);
    }
}
