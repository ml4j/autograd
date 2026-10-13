package org.ml4j.autograd.activation;

import org.ml4j.autograd.arithmetic.ArithmeticOperations;
import org.ml4j.autograd.arithmetic.DifferentiableArithmeticOperations;

import java.util.function.UnaryOperator;

/**
 * Builds differentiable activations from wrapped numerical operations.
 *
 * <p>Scalar and elementwise tensor ReLU use the unary graph helper and arithmetic
 * capabilities for gradient contributions. Wrapped data supplies forward evaluation.
 *
 * @param <V> the differentiable operand and result type
 * @param <D> the wrapped numerical type supporting arithmetic and activations
 * @param <C> the context type, such as tensor shape or Void
 */
public interface DifferentiableActivationOperations<
        V extends DifferentiableArithmeticOperations<V, D, C> & DifferentiableActivationOperations<V, D, C>,
        D extends ArithmeticOperations<D> & ActivationOperations<D>, C>
        extends DifferentiableArithmeticOperations<V, D, C>, ActivationOperations<V> {

    /**
     * Builds a lazy ReLU result with this operand as its single graph input.
     *
     * <p>Preserves context and gradient requirements without changing operand state.
     * Backward multiplies the incoming gradient by the constant mask operand.gt(0).
     * The chosen derivative is one for positive inputs and zero for negative inputs
     * and at zero. Scalar and tensor numerical boundary behaviour follows wrapped data.
     *
     * <p>Mask comparisons are untracked. Higher derivatives retain dependencies in
     * the incoming gradient but do not differentiate the mask; this is a fixed-mask
     * convention, not a claim of smoothness at zero.
     *
     * @return the non-null lazy activation result
     */
    @Override
    default V relu() {
        return applyUnaryOperator(D::relu,
                (incoming, operand) -> incoming.mul(operand.gt(0f)),
                UnaryOperator.identity());
    }
}
