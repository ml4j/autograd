package org.ml4j.autograd.scalar;

import org.ml4j.autograd.Value;

import java.util.function.Supplier;

/**
 * Holds a fixed float scalar with no additional context metadata.
 *
 * <pre>{@code
 * Value<Float, Void> value = new FloatValue(2f);
 * Float scalar = value.data().get(); // 2.0
 * Void context = value.context();   // null: no shape metadata is needed
 * }</pre>
 *
 * <p>The scalar is supplied at construction and remains unchanged. Unlike a tensor
 * value with shape or size context, this scalar uses {@link Void}. All primitive
 * float values are accepted, including signed zero, infinities, and NaN.
 */
public final class FloatValue implements Value<Float, Void> {

    private final float value;

    /**
     * Creates a value holding the supplied scalar.
     *
     * @param value the fixed numerical value
     */
    public FloatValue(float value) {
        this.value = value;
    }

    /**
     * Returns a supplier of the stored scalar.
     *
     * <p>Each successful evaluation returns a non-null {@link Float} representing
     * the same scalar. Supplier and boxed-value object identities are not guaranteed.
     *
     * @return a non-null supplier of the fixed scalar
     */
    @Override
    public Supplier<Float> data() {
        return () -> value;
    }

    /**
     * Returns null because a float scalar needs no additional metadata.
     *
     * <p>A tensor implementation could provide shape context; this implementation
     * uses {@link Void} to express its absence. Reading context does not evaluate data.
     *
     * @return null
     */
    @Override
    public Void context() {
        return null;
    }
}
