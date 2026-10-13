package org.ml4j.autograd.impl;

import org.ml4j.autograd.AutogradValue;
import org.ml4j.autograd.Parameter;

import java.util.Objects;

/**
 * Holds the current autograd leaf through a replaceable reference.
 *
 * <p>An optimizer can install an updated scalar or tensor leaf while the model
 * retains this handle. Existing graphs keep their original leaves.
 * Construction and replacement validate leaves without evaluating numerical data.
 * Numerical and context compatibility, including tensor shape, are the caller's
 * responsibility. Instances are not thread-safe and perform no resource cleanup.
 *
 * @param <V> the concrete autograd value type
 */
public final class ParameterImpl<V extends AutogradValue<V, ?, ?>> implements Parameter<V> {
    private V value;

    /**
     * Creates a handle containing the supplied leaf without evaluating its data.
     * @param value the non-null initial leaf, compatible with the model
     * @throws NullPointerException if value is null
     * @throws IllegalArgumentException if value has graph inputs
     */
    public ParameterImpl(V value) {
        this.value = requireLeaf(value);
    }

    /** {@inheritDoc} */
    @Override
    public V value() {
        return value;
    }

    /** {@inheritDoc} */
    @Override
    public void replace(V value) {
        this.value = requireLeaf(value);
    }

    private static <V extends AutogradValue<V, ?, ?>> V requireLeaf(V value) {
        Objects.requireNonNull(value, "value");
        if (!value.valueNode().inputs().isEmpty()) {
            throw new IllegalArgumentException("Parameter values must be leaves");
        }
        return value;
    }
}
