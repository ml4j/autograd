package org.ml4j.autograd.node;

import org.ml4j.autograd.Accumulatable;
import org.ml4j.autograd.Value;

import java.util.Optional;

/**
 * Holds the current gradient and combines incoming contributions.
 *
 * <p>Scalar contributions of 2 and 3 accumulate to 5. Tensor contributions combine
 * through their own addition operation, with compatible numerical data and context.
 * An absent gradient differs from a calculated gradient whose contents are zero.
 *
 * <p>The node serves as a stable state handle: accumulation and reset change its
 * held reference, not the node itself. Previously returned gradient values and
 * their computation graphs are not mutated by these operations. Graph dependencies
 * and backward rules belong to value nodes, not to this accumulator.
 *
 * @param <V> the gradient value type, providing data access and non-mutating addition
 */
public interface GradNode<V extends Value<?, ?> & Accumulatable<V>> {

    /**
     * Returns the currently held gradient, if one is available.
     *
     * <p>The gradient may be a scalar or a tensor with shape context. An empty
     * result means no gradient is currently held, including after reset.
     *
     * <p>Reading the reference does not evaluate numerical data. Gradient values
     * must be compatible with the forward value they differentiate. A subsequent
     * read may expose a replacement expression; previously returned values do not
     * automatically refresh. No gradient value object-identity guarantee is made.
     *
     * @return a non-null optional containing the gradient, or empty when absent
     */
    Optional<V> gradient();

    /**
     * Adds an incoming contribution to the held gradient.
     *
     * <p>The first scalar or tensor contribution is stored directly. Later
     * contributions are combined using the current value's {@code add(contribution)}.
     *
     * <p>The contribution and resulting sum must be non-null and compatible with
     * the represented gradient's data and context. Addition must not mutate either
     * operand. The node does not directly evaluate numerical data; addition controls
     * any evaluation. If addition throws or returns null, the previously held
     * reference remains unchanged. Gradient expressions may be replaced while
     * this node remains the same handle.
     *
     * @param contribution a non-null compatible gradient contribution
     * @throws NullPointerException if the contribution or resulting sum is null
     */
    void accumulate(V contribution);

    /**
     * Clears the held gradient so that the next contribution starts a fresh sum.
     *
     * <p>After reset, {@link #gradient()} is empty; it does not contain a zero
     * scalar or zero tensor. Resetting an already empty node has the same effect.
     *
     * <p>Only the held reference is cleared. The node identity, forward data,
     * and previously returned gradient values and their graphs remain intact.
     */
    void reset();
}
