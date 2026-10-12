package org.ml4j.autograd.operators;

/**
 * Calculates one operand's gradient contribution for a binary operation.
 *
 * <p>For scalar addition, either operand receives the incoming gradient unchanged.
 * A tensor rule can also adapt the contribution to its operand's shape, for example
 * by reducing broadcast dimensions.
 *
 * <p>The function receives the incoming gradient of the result followed by the
 * forward left and right operands. It returns a non-null contribution compatible
 * with the differentiated operand, without mutating any argument. Accumulation
 * belongs to the caller. Differentiable rules retain the dependencies needed to
 * differentiate the gradient expression itself.
 *
 * @param <V> the type representing operands and gradient values
 */
@FunctionalInterface
public interface BinaryGradient<V> {

    /**
     * Calculates the contribution associated with this operand's derivative rule.
     *
     * <p>Operand order is unchanged even when both positions reference the same
     * scalar or tensor value. This rule does not receive the forward result value;
     * the caller has already extracted its incoming gradient.
     *
     * @param incomingGradient the non-null gradient of the operation's result
     * @param leftOperand the non-null forward left operand
     * @param rightOperand the non-null forward right operand
     * @return a non-null contribution compatible with the differentiated operand's data/context
     */
    V apply(V incomingGradient, V leftOperand, V rightOperand);
}
