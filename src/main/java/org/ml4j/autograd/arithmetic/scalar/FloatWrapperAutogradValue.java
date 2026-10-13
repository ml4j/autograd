package org.ml4j.autograd.arithmetic.scalar;

import org.ml4j.autograd.AutogradValue;
import org.ml4j.autograd.activation.DifferentiableActivationOperations;
import org.ml4j.autograd.arithmetic.DifferentiableArithmeticOperations;

/**
 * A differentiable scalar whose numerical data implements FloatWrapper arithmetic.
 *
 * <p>Data reads return wrappers; floatValue() on those wrappers reads the primitive
 * scalar. Void context means no shape metadata is needed. Shared arithmetic and
 * activation defaults supply operation rules; implementations supply graph machinery.
 */
public interface FloatWrapperAutogradValue extends AutogradValue<FloatWrapperAutogradValue, FloatWrapper, Void>, DifferentiableArithmeticOperations<FloatWrapperAutogradValue, FloatWrapper, Void>, DifferentiableActivationOperations<FloatWrapperAutogradValue, FloatWrapper, Void> {
}
