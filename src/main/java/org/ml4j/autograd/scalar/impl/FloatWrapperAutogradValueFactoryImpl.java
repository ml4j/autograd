package org.ml4j.autograd.scalar.impl;

import org.ml4j.autograd.scalar.FloatWrapper;
import org.ml4j.autograd.scalar.FloatWrapperAutogradValue;
import org.ml4j.autograd.scalar.FloatWrapperAutogradValueFactory;

import java.util.function.Supplier;

/**
 * Creates wrapped scalar leaves using the existing scalar value constructors.
 *
 * <pre>{@code
 * FloatWrapperAutogradValueFactory factory = new FloatWrapperAutogradValueFactoryImpl();
 * var x = factory.create(3f, true);
 * x.mul(x).backward();
 * float gradient = x.gradNode().gradient().orElseThrow().data().get().floatValue(); // 6
 * }</pre>
 *
 * <p>Supplier creation is lazy and adds no caching: suppliers must return stable
 * numerical contents on every successful read. This stateless factory does not
 * construct operation-result graphs or alter existing values and gradients.
 */
public final class FloatWrapperAutogradValueFactoryImpl implements FloatWrapperAutogradValueFactory {
    /** Creates a stateless scalar leaf factory. */
    public FloatWrapperAutogradValueFactoryImpl() {}

    /** {@inheritDoc} */
    @Override
    public FloatWrapperAutogradValue create(float value, boolean requiresGrad) {
        return new FloatWrapperAutogradValueImpl(value, requiresGrad);
    }

    /**
     * Creates a fresh leaf without evaluating or caching the numerical supplier.
     * @param data the non-null supplier of stable, non-null numerical data
     * @param context null, because scalar Void context carries no metadata
     * @param requiresGrad whether the new leaf requires library gradients
     * @return a new scalar leaf with null context
     * @throws NullPointerException if data is null
     */
    @Override
    public FloatWrapperAutogradValue create(Supplier<FloatWrapper> data, Void context, boolean requiresGrad) {
        return new FloatWrapperAutogradValueImpl(data, requiresGrad);
    }
}
