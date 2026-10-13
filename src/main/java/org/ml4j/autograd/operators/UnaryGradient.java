package org.ml4j.autograd.operators;

/**
 * Calculates an operand's gradient contribution for a unary operation.
 *
 * <p>Scalar or elementwise tensor negation negates the incoming gradient. Other
 * rules can use the original operand to retain dependencies for higher derivatives.
 * Contributions are non-null and compatible with the operand's data and context;
 * rules do not mutate arguments or accumulate gradient state.
 *
 * @param <V> the type representing operands and gradient values
 */
@FunctionalInterface
public interface UnaryGradient<V> {
    /**
     * Calculates the contribution from the incoming gradient and forward operand.
     *
     * <p>The caller extracts the gradient from the result; the result value itself
     * is not passed to this rule. Accumulation belongs to the caller.
     *
     * @param incomingGradient the non-null gradient of the operation's result
     * @param operand the non-null original forward operand
     * @return a non-null compatible contribution
     */
    V apply(V incomingGradient, V operand);
}
