package org.ml4j.autograd.arithmetic;

import org.ml4j.autograd.AutogradValue;
import org.ml4j.autograd.operators.BinaryGradient;

import java.util.function.BinaryOperator;

/**
 * Defines differentiable operations by delegating forward arithmetic to wrapped data.
 *
 * <p>A scalar wrapper supplies add and mul on its numerical values. A tensor
 * adapter can supply numerical operations and context mapping while reusing graph
 * construction supplied by the autograd implementation.
 *
 * <p>The current default addition passes gradients unchanged. Tensor implementations
 * requiring broadcast reduction must adapt those contributions explicitly. This
 * interface supplies addition and ordinary product-rule multiplication by default.
 * Tensor types with broadcasting or non-elementwise multiplication must override
 * the relevant defaults to provide appropriate shape and derivative rules.
 *
 * @param <V> the differentiable operand and result type
 * @param <D> the wrapped numerical type implementing arithmetic
 * @param <C> the context type, such as shape or Void
 */
public interface DifferentiableArithmeticOperations<V extends AutogradValue<V, D, C> & DifferentiableArithmeticOperations<V, D, C>, D extends ArithmeticOperations<D>, C> extends AutogradValue<V, D, C>, ArithmeticOperations<V> {

    /**
     * Builds a lazy addition result with both operands recorded as inputs.
     *
     * <p>Scalar arithmetic or tensor addition is supplied by the concrete type.
     * Forward construction leaves operand data and gradient state unchanged.
     * Each tracked operand receives the result's incoming gradient unchanged when
     * the result node's local backward callback is invoked.
     * This default assumes contribution shapes already match the operands; tensor
     * implementations with broadcasting must override it to reduce gradients appropriately.
     *
     * @param other the right operand
     * @return the new forward sum
     * @throws NullPointerException if other is null
     */
    @Override
    default V add(V other) {
        return applyBinaryOperator(other, D::add,
                (incomingGradient, leftOperand, rightOperand) -> incomingGradient,
                (incomingGradient, leftOperand, rightOperand) -> incomingGradient,
                this::mapContext);
    }

    /**
     * Builds a lazy product using the wrapped numerical data's multiplication.
     *
     * <p>Scalar values 2 and 3 produce 6. The left contribution is incoming-gradient
     * times the right operand; the right contribution is incoming-gradient times the
     * left operand. This also describes same-shape elementwise tensor multiplication,
     * but not matrix multiplication or broadcast-gradient reductions.
     *
     * <p>Neither operand is mutated. The result records ordered inputs and combined
     * gradient requirements, and gradient operations retain higher-derivative graphs.
     * Implementations with different numerical semantics must override this default.
     *
     * @param other the non-null compatible right operand
     * @return the lazy differentiable product
     * @throws NullPointerException if other is null
     */
    @Override
    default V mul(V other) {
        return applyBinaryOperator(other, D::mul,
                (incomingGradient, leftOperand, rightOperand) -> incomingGradient.mul(rightOperand),
                (incomingGradient, leftOperand, rightOperand) -> incomingGradient.mul(leftOperand),
                this::mapContext);
    }

    /**
     * Maps operand metadata to result metadata.
     *
     * <p>Scalars use null context; a tensor implementation can calculate output shape.
     * Context mapping does not evaluate numerical data.
     *
     * @param leftContext left operand context
     * @param rightContext right operand context
     * @return compatible output context, or null when no metadata is needed
     */
    C mapContext(C leftContext, C rightContext);

    /**
     * Builds a binary result and installs its local backward callback.
     *
     * <p>Default arithmetic passes a numerical method reference such as D::add and
     * two derivative rules. The result callback receives the forward result value;
     * derivative rules receive its incoming gradient and the ordered operands.
     *
     * <p>Implementations map context once, defer numerical evaluation, preserve
     * ordered inputs including duplicates, and combine operand gradient requirements.
     * Local propagation skips absent gradients and constants and accumulates each
     * tracked operand's contribution without traversing the graph.
     *
     * @param other the non-null right operand
     * @param forward the non-null numerical operation
     * @param leftGradient the non-null left operand contribution rule
     * @param rightGradient the non-null right operand contribution rule
     * @param contextMapper the non-null output context mapper
     * @return the differentiable binary result
     * @throws NullPointerException if an argument is null
     */
    V applyBinaryOperator(V other, BinaryOperator<D> forward,
                          BinaryGradient<V> leftGradient, BinaryGradient<V> rightGradient,
                          BinaryOperator<C> contextMapper);
}
