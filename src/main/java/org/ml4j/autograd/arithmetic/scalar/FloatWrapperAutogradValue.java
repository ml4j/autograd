package org.ml4j.autograd.arithmetic.scalar;

import org.ml4j.autograd.AutogradValue;
import org.ml4j.autograd.arithmetic.DifferentiableArithmeticOperations;

/**
 * A differentiable scalar whose numerical data implements FloatWrapper arithmetic.
 *
 * <p>Data reads return wrappers; floatValue() on those wrappers reads the primitive
 * scalar. Void context means no shape metadata is needed. Addition uses the shared
 * arithmetic default, while implementations provide multiplication and graph machinery.
 */
public interface FloatWrapperAutogradValue extends AutogradValue<FloatWrapperAutogradValue, FloatWrapper, Void>, DifferentiableArithmeticOperations<FloatWrapperAutogradValue, FloatWrapper, Void> {
}
