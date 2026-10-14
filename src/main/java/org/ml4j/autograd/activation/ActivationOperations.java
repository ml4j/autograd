package org.ml4j.autograd.activation;

/**
 * Supplies non-mutating numerical activation functions for scalars or tensors.
 *
 * <p>Activations are a separate capability from arithmetic and graph construction.
 * A scalar applies an activation to its number; a tensor applies it elementwise.
 *
 * @param <V> the concrete activation result type
 */
public interface ActivationOperations<V extends ActivationOperations<V>> {
    /**
     * Returns the rectified linear activation, max(0, x).
     *
     * <p>For example, -2 becomes zero and 3 remains 3. Tensor shape is preserved.
     * The operand's data, graph, and gradient state are unchanged. Numerical
     * implementations document non-finite and signed-zero behaviour.
     *
     * @return a non-null activation result
     */
    V relu();

    /**
     * Returns the logistic sigmoid, 1 / (1 + exp(-x)).
     *
     * <p>Zero becomes 0.5; tensors apply the function elementwise and preserve shape.
     * The operand's data, graph, and gradient state are unchanged. Implementations
     * document rounding and non-finite behaviour.
     *
     * @return a non-null activation result
     */
    V sigmoid();
}
