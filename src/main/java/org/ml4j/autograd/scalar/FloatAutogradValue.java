package org.ml4j.autograd.scalar;

import org.ml4j.autograd.AutogradValue;

/**
 * A differentiable scalar whose numerical data is a Java Float.
 *
 * <p>Read a scalar with {@code value.data().get()}. Arithmetic is implemented
 * directly over float values, with lazy graph results and Void context. Unlike
 * {@link FloatWrapperAutogradValue}, this type does not compose wrapped numerical
 * arithmetic, activation or math capabilities. Existing values are not mutated.
 *
 * @see org.ml4j.autograd.scalar.impl.FloatAutogradValueImpl
 * @see FloatAutogradValueFactory
 */
public interface FloatAutogradValue extends AutogradValue<FloatAutogradValue, Float, Void> {

    /** Adds primitive scalar data through the generic differentiable operation machinery. */
    FloatAutogradValue add(FloatAutogradValue other);

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
    FloatAutogradValue sub(FloatAutogradValue other);

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
    FloatAutogradValue mul(FloatAutogradValue other);

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
    FloatAutogradValue div(FloatAutogradValue other);

    /**
     * Returns this scalar's negation, for example 2 becomes -2.
     *
     * <p>Evaluation is deferred and context is unchanged, as for elementwise tensor
     * negation. The result records one input and preserves the gradient requirement.
     * The backward rule negates the incoming gradient without mutating either value.
     *
     * @return the lazy differentiable negation
     */
    FloatAutogradValue neg();
}
