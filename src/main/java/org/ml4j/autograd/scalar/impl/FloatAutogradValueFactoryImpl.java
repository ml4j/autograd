package org.ml4j.autograd.scalar.impl;

import org.ml4j.autograd.scalar.FloatAutogradValue;
import org.ml4j.autograd.scalar.FloatAutogradValueFactory;
import java.util.function.Supplier;

/**
 * Creates raw scalar leaves through FloatAutogradValueImpl constructors.
 * <pre>{@code
 * FloatAutogradValueFactory factory = new FloatAutogradValueFactoryImpl();
 * var x = factory.create(3f, true);
 * x.mul(x).backward(); // x's gradient contains 6f
 * }</pre>
 * <p>Stateless construction preserves lazy non-cached supplier semantics. Supplier
 * data must remain stable; this factory does not build operation-result graphs.
 */
public final class FloatAutogradValueFactoryImpl implements FloatAutogradValueFactory {
    /** Creates a stateless raw scalar leaf factory. */
    public FloatAutogradValueFactoryImpl() {}

    /** {@inheritDoc} */
    @Override
    public FloatAutogradValue create(float value, boolean requiresGrad) {
        return new FloatAutogradValueImpl(value, requiresGrad);
    }

    /**
     * Creates a fresh leaf without evaluating or caching its supplier.
     * @param data the non-null supplier of stable, non-null Float data
     * @param context null, because scalars need no context metadata
     * @param requiresGrad whether library gradients are required
     * @return a new leaf with null context
     * @throws NullPointerException if data is null
     */
    @Override
    public FloatAutogradValue create(Supplier<Float> data, Void context, boolean requiresGrad) {
        return new FloatAutogradValueImpl(data, requiresGrad);
    }
}
