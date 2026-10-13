package org.ml4j.autograd;

import java.util.function.Supplier;

/**
 * Creates autograd leaves with an explicit choice of whether to calculate gradients.
 *
 * <p>A scalar factory can accept a supplier of Float data with Void context. A
 * tensor factory can accept a supplier of backend tensor data with shape context.
 * Operation-result construction with graph inputs is a separate responsibility.
 *
 * @param <V> the concrete autograd value type
 * @param <D> the numerical data type
 * @param <C> the context type, such as shape or Void
 */
@FunctionalInterface
public interface AutogradValueFactory<V extends AutogradValue<V, D, C>, D, C> {

    /**
     * Creates a new leaf without evaluating its numerical data.
     *
     * <p>The leaf has no graph inputs, no backward callback, and an initially absent
     * gradient. Its gradient requirement matches requiresGrad. Supplied data and
     * context must satisfy Value's stability and compatibility contract; successful
     * data evaluations must return non-null numerical contents. Caching remains an
     * implementation choice. Creation does not change existing values or gradients.
     *
     * @param data a non-null supplier of stable, non-null numerical data
     * @param context stable metadata compatible with data, or null when none is needed
     * @param requiresGrad whether differentiation should calculate this leaf's gradient
     * @return a new, non-null autograd leaf
     * @throws NullPointerException if data is null
     */
    V create(Supplier<D> data, C context, boolean requiresGrad);
}
