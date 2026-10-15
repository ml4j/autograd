package org.ml4j.autograd.scalar;

import org.ml4j.autograd.scalar.impl.FloatWrapperAutogradValueFactoryImpl;

import org.ml4j.autograd.AutogradValueFactory;

/**
 * Creates wrapped scalar leaves with explicit library gradient tracking.
 *
 * <p>Binds numerical data to FloatWrapper and context to Void, so clients need not
 * repeat the generic arguments. Like tensor leaf factories, it provides inherited
 * supplier-based creation while leaving operation-result construction to values.
 *
 * @see FloatWrapperAutogradValueFactoryImpl
 */
public interface FloatWrapperAutogradValueFactory
        extends AutogradValueFactory<FloatWrapperAutogradValue, FloatWrapper, Void> {
    /**
     * Creates a fresh scalar leaf from a fixed primitive value.
     *
     * <p>The leaf has no graph inputs or backward callback and its gradient is
     * initially absent. All float values are accepted; context is null.
     *
     * @param value the fixed numerical scalar
     * @param requiresGrad whether backward passes calculate the leaf's gradient
     * @return a new scalar leaf
     */
    FloatWrapperAutogradValue create(float value, boolean requiresGrad);
}
