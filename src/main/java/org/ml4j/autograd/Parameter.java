package org.ml4j.autograd;

/**
 * Provides a stable parameter handle whose current autograd leaf can be replaced.
 *
 * <p>An optimizer can replace a scalar weight of 2 with a new leaf holding 1.9,
 * or replace a tensor weight with updated numerical data of compatible shape.
 * The model retains the same parameter handle and reads value() to build each
 * forward graph. Each Value keeps stable numerical contents throughout its lifetime.
 *
 * <p>Graphs capture the leaf returned by value(), not a live lookup through this
 * handle. Replacement leaves earlier values and their graphs unchanged; future
 * reads select the replacement. This interface provides no arithmetic, gradient
 * accumulation, optimizer algorithm, or native-resource cleanup.
 *
 * <p>Current values are non-null leaves with no graph inputs. Callers supply
 * replacements compatible with the model's numerical type and context, such as
 * tensor shape. Compatibility validation is implementation-defined. Use sequentially;
 * concurrent replacement and graph construction are unsupported.
 *
 * @param <V> the concrete autograd value type held by this parameter
 */
public interface Parameter<V extends AutogradValue<V, ?, ?>> {

    /**
     * Returns the current leaf without evaluating numerical data.
     *
     * <p>Repeated reads return the same leaf until replacement. Read this method
     * when building a forward graph; previously returned leaves do not refresh.
     *
     * @return the current non-null autograd leaf
     */
    V value();

    /**
     * Selects the leaf used by subsequent reads without evaluating numerical data.
     *
     * <p>Replaces only the held reference. Neither leaf's data, context, graph,
     * gradient requirement, or stored gradient is changed. The supplied leaf's
     * existing gradient state is preserved; use a fresh leaf for a fresh training
     * step. Passing the current leaf is permitted. Invalid replacements leave the
     * current reference unchanged.
     *
     * @param value a non-null autograd leaf compatible with the model
     * @throws NullPointerException if value is null
     * @throws IllegalArgumentException if value has graph inputs
     */
    void replace(V value);
}
