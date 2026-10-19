package org.ml4j.autograd;

/**
 * Provides a stable parameter handle whose current value can be replaced.
 *
 * <p>An optimizer can replace a scalar weight of 2 with a new leaf holding 1.9,
 * or replace a tensor weight with updated numerical data of compatible shape.
 * The model retains the same parameter handle and reads value() to build each
 * forward graph. Each Value keeps stable numerical contents throughout its lifetime.
 *
 * <p>Graphs capture the selected autograd leaf, not a live lookup through this
 * handle. Replacement leaves earlier values and their graphs unchanged; future
 * reads select the replacement. This interface provides no arithmetic, gradient
 * accumulation, optimizer algorithm, or native-resource cleanup.
 *
 * <p>Current values are non-null; implementations define leaf validation. Callers supply
 * replacements compatible with the model's numerical type and context, such as
 * tensor shape. Compatibility validation is implementation-defined. Use sequentially;
 * concurrent replacement and graph construction are unsupported.
 *
 * @param <V> the value or wrapper type held by this parameter
 */
public interface Parameter<V> {

    /**
     * Returns the current value without evaluating numerical data.
     *
     * <p>Repeated reads return the same value until replacement. Read this method
     * when building a forward graph; previously returned leaves do not refresh.
     *
     * @return the current non-null value
     */
    V value();

    /**
     * Selects the value used by subsequent reads without evaluating numerical data.
     *
     * <p>Replaces only the held reference. Neither value's data, context, graph,
     * gradient requirement, or stored gradient is changed. The supplied value's
     * existing gradient state is preserved; use a fresh leaf for a fresh training
     * step. Passing the current value is permitted. Invalid replacements leave the
     * current reference unchanged.
     *
     * @param value a non-null value compatible with the model
     * @throws NullPointerException if value is null
     * @throws IllegalArgumentException if value fails implementation-specific validation
     */
    void replace(V value);
}
