package org.ml4j.autograd.impl;

import org.ml4j.autograd.AutogradValue;
import org.ml4j.autograd.Parameter;

import java.util.Objects;
import java.util.function.Function;

/**
 * Holds the current autograd leaf through a replaceable reference.
 *
 * <p>An optimizer can install an updated scalar or tensor leaf while the model
 * retains this handle. Existing graphs keep their original leaves.
 * Construction and replacement validate leaves without evaluating numerical data.
 * Numerical and context compatibility, including tensor shape, are the caller's
 * responsibility. Instances are not thread-safe and perform no resource cleanup.
 *
 * @param <V> the value or wrapper type
 */
public final class ParameterImpl<V> implements Parameter<V> {
    private V value;
    private final Function<? super V, ? extends AutogradValue<?, ?, ?>> extractor;

    /**
     * Creates a handle for a direct autograd leaf.
     *
     * <p>Construction and later replacements require values implementing AutogradValue.
     * This is checked at runtime because V also supports wrappers.
     *
     * @param value the non-null initial autograd leaf
     * @throws NullPointerException if value is null
     * @throws IllegalArgumentException if value is not an autograd value or has graph inputs
     */
    public ParameterImpl(V value) {
        this(value, candidate -> {
            if (candidate instanceof AutogradValue<?, ?, ?> autogradValue) {
                return autogradValue;
            }
            throw new IllegalArgumentException("Parameter values must implement AutogradValue");
        });
    }

    /**
     * Creates a handle whose values expose an autograd leaf through an extractor.
     *
     * <p>For example, a tensor wrapper can supply Tensor::autogradValue. The extractor
     * must return its stable leaf without evaluating numerical data or mutating state.
     * It runs once per non-null candidate on construction and replacement, never on reads.
     * Extractor exceptions propagate; validation failure preserves the held reference,
     * but cannot roll back extractor side effects.
     *
     * @param value the non-null initial value or wrapper
     * @param extractor extracts the non-null represented autograd leaf
     * @throws NullPointerException if value, extractor or extracted leaf is null
     * @throws IllegalArgumentException if the extracted value has graph inputs
     */
    public ParameterImpl(V value,
            Function<? super V, ? extends AutogradValue<?, ?, ?>> extractor) {
        this.extractor = Objects.requireNonNull(extractor, "extractor");
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

    private V requireLeaf(V value) {
        Objects.requireNonNull(value, "value");
        var leaf = Objects.requireNonNull(extractor.apply(value), "extracted value");
        if (!leaf.valueNode().inputs().isEmpty()) {
            throw new IllegalArgumentException("Parameter values must be leaves");
        }
        return value;
    }
}
