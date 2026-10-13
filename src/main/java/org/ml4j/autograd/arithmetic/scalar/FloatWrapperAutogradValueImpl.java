package org.ml4j.autograd.arithmetic.scalar;

import org.ml4j.autograd.impl.AutogradValueImpl;
import org.ml4j.autograd.node.ValueNode;

import java.util.List;
import java.util.function.Supplier;

/**
 * Demonstrates shared differentiable arithmetic over immutable wrapped float data.
 *
 * <pre>{@code
 * var first = new FloatWrapperAutogradValueImpl(2f, true);
 * var second = new FloatWrapperAutogradValueImpl(3f, false);
 * var sum = first.add(second);
 * FloatWrapper data = sum.data().get();
 * float scalar = data.getFloat(); // 5.0
 * }</pre>
 *
 * <p>Uses Void context because scalars need no shape metadata. Unlike a tensor
 * subclass, numerical arithmetic is delegated to FloatWrapper. Forward results are lazy;
 * supplied numerical data must remain stable. Addition and multiplication support
 * local and graph-wide propagation; the implicit scalar seed is one.
 */
public final class FloatWrapperAutogradValueImpl extends AutogradValueImpl<FloatWrapperAutogradValue, FloatWrapper, Void> implements FloatWrapperAutogradValue {

    /**
     * Creates a scalar leaf from fixed numerical data.
     * @param value the non-null immutable scalar wrapper
     * @param requiresGrad whether a backward pass should calculate its gradient
     */
    public FloatWrapperAutogradValueImpl(FloatWrapper value, boolean requiresGrad) {
        this(() -> value, requiresGrad);
    }

    /**
     * Creates a scalar leaf from fixed numerical data.
     * @param value the scalar, including any valid primitive float value
     * @param requiresGrad whether a backward pass should calculate its gradient
     */
    public FloatWrapperAutogradValueImpl(float value, boolean requiresGrad) {
        this(() -> new FloatWrapperImpl(value), requiresGrad);
    }


    /**
     * Creates a scalar leaf with deferred numerical data.
     * @param data a supplier of stable, non-null FloatWrapper data
     * @param requiresGrad whether a backward pass should calculate its gradient
     * @throws NullPointerException if the supplier is null
     */
    public FloatWrapperAutogradValueImpl(Supplier<FloatWrapper> data, boolean requiresGrad) {
        this(data, requiresGrad, List.of());
    }

    private FloatWrapperAutogradValueImpl(Supplier<FloatWrapper> data, boolean requiresGrad, List<? extends ValueNode<?>> inputs) {
        super(data, null, requiresGrad, inputs);
    }

    /** Returns this scalar as the node's concrete owner. */
    @Override protected FloatWrapperAutogradValue self() { return this; }

    /** Supplies a constant scalar seed of one for implicit backward execution. */
    @Override protected FloatWrapperAutogradValue defaultGradient() {
        return new FloatWrapperAutogradValueImpl(1f, false);
    }

    /** Scalars need no context metadata. */
    @Override public Void mapContext(Void leftContext, Void rightContext) { return null; }

    /** Constructs the lazy scalar result with its ordered operands and combined gradient requirement. */
    @Override
    protected FloatWrapperAutogradValue createAutogradValue(Supplier<FloatWrapper> dataSupplier, Void context,
            boolean requiresGrad, List<? extends ValueNode<?>> inputs) {
        return new FloatWrapperAutogradValueImpl(dataSupplier, requiresGrad, inputs);
    }
}
