package org.ml4j.autograd.scalar;

import org.ml4j.autograd.AutogradValueFactory;

/**
 * Creates raw Float scalar leaves with explicit gradient tracking.
 * <p>Binds the generic data/context types to Float and Void. Supplier creation
 * follows AutogradValueFactory; operation-result construction belongs to values.
 * @see org.ml4j.autograd.scalar.impl.FloatAutogradValueFactoryImpl
 */
public interface FloatAutogradValueFactory
        extends AutogradValueFactory<FloatAutogradValue, Float, Void> {
    /**
     * Creates a new leaf with null context, no inputs/callback and absent gradient.
     * @param value the fixed scalar, including any primitive float value
     * @param requiresGrad whether library gradients are required
     * @return a new scalar leaf
     */
    FloatAutogradValue create(float value, boolean requiresGrad);
}
