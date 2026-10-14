package org.ml4j.autograd.math;

import org.ml4j.autograd.arithmetic.ArithmeticOperations;
import org.ml4j.autograd.arithmetic.DifferentiableArithmeticOperations;
import java.util.function.UnaryOperator;

/**
 * Builds lazy mathematical expressions using wrapped numerical functions.
 *
 * <p>Each result records one operand, preserves context (such as tensor shape)
 * and gradient requirements, and leaves operand data and gradients unchanged.
 * Differentiable gradient contributions retain higher-derivative dependencies.
 * Numerical domains, rounding and non-finite behaviour follow wrapped data and
 * arithmetic; singular derivatives are not regularised.
 *
 * @param <V> the differentiable operand and result type
 * @param <D> the wrapped numerical arithmetic and math type
 * @param <C> the context type, such as tensor shape or Void
 */
public interface DifferentiableMathOperations<
        V extends DifferentiableArithmeticOperations<V, D, C> & DifferentiableMathOperations<V, D, C>,
        D extends ArithmeticOperations<D> & MathOperations<D>, C>
        extends DifferentiableArithmeticOperations<V, D, C>, MathOperations<V> {

    /** Returns a lazy exponential with backward contribution incoming * exp(operand). */
    @Override
    default V exp() {
        return applyUnaryOperator(D::exp,
                (incoming, operand) -> incoming.mul(operand.exp()), UnaryOperator.identity());
    }

    /** Returns a lazy logarithm with backward contribution incoming / operand. */
    @Override
    default V log() {
        return applyUnaryOperator(D::log,
                (incoming, operand) -> incoming.div(operand), UnaryOperator.identity());
    }

    /**
     * Returns a lazy power with contribution incoming * operand.pow(exponent - 1) * exponent.
     *
     * <p>The exponent is untracked. The general rule is applied even for zero and
     * one: singular intermediates may produce NaN (for example pow(0) at zero).
     * Higher derivatives retain the same rule and backend boundary limitations.
     * @param exponent the fixed float power
     * @return a non-null lazy power result
     */
    @Override
    default V pow(float exponent) {
        return applyUnaryOperator(data -> data.pow(exponent),
                (incoming, operand) -> incoming.mul(operand.pow(exponent - 1f)).mul(exponent),
                UnaryOperator.identity());
    }
}
