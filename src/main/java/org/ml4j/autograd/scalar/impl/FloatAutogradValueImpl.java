package org.ml4j.autograd.scalar.impl;

import org.ml4j.autograd.scalar.FloatAutogradValue;

import org.ml4j.autograd.impl.AutogradValueImpl;
import org.ml4j.autograd.node.ValueNode;

import java.util.List;
import java.util.function.Supplier;

/**
 * Implements raw Float scalar arithmetic over the shared autograd structure.
 *
 * <pre>{@code
 * var first = new FloatAutogradValueImpl(2f, true);
 * var second = new FloatAutogradValueImpl(3f, false);
 * var sum = first.add(second);
 * Float data = sum.data().get(); // 5.0
 * }</pre>
 *
 * <p>Uses Void context because scalars need no shape metadata. Unlike a tensor
 * subclass, numerical addition is ordinary float addition. Forward results are lazy;
 * supplied numerical data must remain stable. Arithmetic operations support
 * local and graph-wide propagation; the implicit scalar seed is one.
 */
public final class FloatAutogradValueImpl extends AutogradValueImpl<FloatAutogradValue, Float, Void> implements FloatAutogradValue {

    /** Adds primitive scalar data through the generic differentiable operation machinery. */
    @Override
    public FloatAutogradValue add(FloatAutogradValue other) {
        return applyBinaryOperator(other, (leftData, rightData) -> leftData + rightData,
                (incomingGradient, leftOperand, rightOperand) -> incomingGradient,
                (incomingGradient, leftOperand, rightOperand) -> incomingGradient,
                this::mapContext);
    }

    /**
     * Returns the difference, for example 5 minus 3 equals 2.
     *
     * <p>Composes lazy negation and addition without mutating either operand.
     * As with compatible elementwise tensor subtraction, backward contributes the
     * incoming gradient to the left operand and its negation to the right.
     * Existing graph requirements and higher-derivative rules are preserved.
     *
     * @param other the non-null right scalar operand
     * @return the lazy difference represented by negation and addition nodes
     * @throws NullPointerException if other is null
     */
    @Override
    public FloatAutogradValue sub(FloatAutogradValue other) {
        return add(other.neg());
    }

    /**
     * Returns the product of this scalar and another scalar.
     *
     * <p>Multiplying 2 by 3 produces 6. With an incoming gradient of one, the left
     * operand receives a contribution of 3 and the right receives 2.
     *
     * <p>Numerical evaluation is deferred. The result records both operands in order
     * and requires gradients if either operand does. Neither operand is mutated.
     * Contributions are incoming-gradient times the opposite operand, retaining
     * differentiable graphs for higher derivatives. Repeated operands contribute twice.
     *
     * @param other the non-null right scalar operand
     * @return the lazy multiplication result
     * @throws NullPointerException if other is null
     */
    @Override
    public FloatAutogradValue mul(FloatAutogradValue other) {
        return applyBinaryOperator(other, (left, right) -> left * right,
                (incomingGradient, leftOperand, rightOperand) -> incomingGradient.mul(rightOperand),
                (incomingGradient, leftOperand, rightOperand) -> incomingGradient.mul(leftOperand),
                this::mapContext);
    }

    /**
     * Returns the lazy quotient, for example 6 divided by 2 is 3.
     *
     * <p>As with compatible elementwise tensors, contributions are incoming-gradient
     * divided by denominator and -(incoming-gradient times numerator) divided by
     * denominator squared. Differentiable operations preserve higher derivatives.
     * Neither operand is mutated; inputs and gradient requirements follow binary construction.
     * Float zero and non-finite cases follow IEEE arithmetic, including in derivative rules.
     *
     * @param other the non-null denominator
     * @return the lazy differentiable quotient
     * @throws NullPointerException if other is null
     */
    @Override
    public FloatAutogradValue div(FloatAutogradValue other) {
        return applyBinaryOperator(other, (left, right) -> left / right,
                (incomingGradient, leftOperand, rightOperand) -> incomingGradient.div(rightOperand),
                (incomingGradient, leftOperand, rightOperand) ->
                        incomingGradient.mul(leftOperand).neg().div(rightOperand.mul(rightOperand)),
                this::mapContext);
    }

    /**
     * Returns this scalar's negation, for example 2 becomes -2.
     *
     * <p>Evaluation is deferred and context is unchanged, as for elementwise tensor
     * negation. The result records one input and preserves the gradient requirement.
     * The backward rule negates the incoming gradient without mutating either value.
     *
     * @return the lazy differentiable negation
     */
    @Override
    public FloatAutogradValue neg() {
        return applyUnaryOperator(value -> -value,
                (incomingGradient, operand) -> incomingGradient.neg(), context -> context);
    }

    /**
     * Creates a scalar leaf from fixed numerical data.
     * @param value the scalar, including any valid primitive float value
     * @param requiresGrad whether a future backward pass should calculate its gradient
     */
    public FloatAutogradValueImpl(float value, boolean requiresGrad) {
        this(() -> value, requiresGrad);
    }

    /**
     * Creates a scalar leaf with deferred numerical data.
     * @param data a supplier of stable, non-null float data
     * @param requiresGrad whether a future backward pass should calculate its gradient
     * @throws NullPointerException if the supplier is null
     */
    public FloatAutogradValueImpl(Supplier<Float> data, boolean requiresGrad) {
        this(data, requiresGrad, List.of());
    }

    private FloatAutogradValueImpl(Supplier<Float> data, boolean requiresGrad, List<? extends ValueNode<?>> inputs) {
        super(data, null, requiresGrad, inputs);
    }

    /** Returns this scalar as the node's concrete owner. */
    @Override protected FloatAutogradValue self() { return this; }

    /** Supplies a constant scalar seed of one for implicit backward execution. */
    @Override protected FloatAutogradValue defaultGradient() {
        return new FloatAutogradValueImpl(1f, false);
    }

    /** Scalars need no context metadata. */
    @Override public Void mapContext(Void left, Void right) { return null; }

    /** Constructs the lazy scalar result with its ordered operands and combined gradient requirement. */
    @Override
    protected FloatAutogradValue createAutogradValue(Supplier<Float> dataSupplier, Void context,
            boolean requiresGrad, List<? extends ValueNode<?>> inputs) {
        return new FloatAutogradValueImpl(dataSupplier, requiresGrad, inputs);
    }
}
