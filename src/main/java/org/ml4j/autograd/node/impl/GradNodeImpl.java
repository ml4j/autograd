package org.ml4j.autograd.node.impl;

import org.ml4j.autograd.Accumulatable;
import org.ml4j.autograd.Value;
import org.ml4j.autograd.node.GradNode;

import java.util.Objects;
import java.util.Optional;

/**
 * Stores a gradient reference and combines contributions through their addition operation.
 *
 * <p>Scalar contributions of 2 and 3 produce a held gradient of 5. Tensor values
 * can provide their own compatible addition; this class needs no tensor-specific code.
 *
 * <p>Starts empty. Accumulation replaces the held reference only after addition
 * succeeds with a non-null result. Reset clears that reference without mutating
 * earlier gradient values. Data evaluation and compatibility checks belong to
 * the value's addition operation. Instances are not thread-safe.
 *
 * @param <V> the gradient value type, supporting non-mutating addition
 */
public final class GradNodeImpl<V extends Value<?, ?> & Accumulatable<V>> implements GradNode<V> {

    private V gradient;

    /** Creates an empty gradient-state handle without evaluating numerical data. */
    public GradNodeImpl() {
    }

    /** {@inheritDoc} */
    @Override
    public Optional<V> gradient() {
        return Optional.ofNullable(gradient);
    }

    /** {@inheritDoc} */
    @Override
    public void accumulate(V contribution) {
        Objects.requireNonNull(contribution, "contribution");
        // Compute and validate before assignment so failed addition preserves the held reference.
        V combined = gradient == null ? contribution : gradient.add(contribution);
        gradient = Objects.requireNonNull(combined, "combined gradient");
    }

    /** {@inheritDoc} */
    @Override
    public void reset() {
        gradient = null;
    }
}
