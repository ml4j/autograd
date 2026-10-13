package org.ml4j.autograd.arithmetic.scalar;

/**
 * Stores a fixed float and returns new wrappers for arithmetic results.
 *
 * <p>Wrappers of 2 and 3 sum to 5 or multiply to 6. This scalar data type has no
 * graph or gradient state; tensor data implementations can provide a similar API.
 * All primitive float values are accepted. The stored value cannot be replaced.
 */
public class FloatWrapperImpl implements FloatWrapper {

    private final float value;

    /**
     * Creates fixed scalar numerical data.
     * @param value the primitive float to wrap
     */
    public FloatWrapperImpl(float value) {
        this.value = value;
    }

    /** {@inheritDoc} */
    @Override
    public float floatValue() {
        return value;
    }

    /** {@inheritDoc} */
    @Override
    public FloatWrapper add(FloatWrapper other) {
        return createFloatWrapper(
                this.floatValue() + other.floatValue());
    }

    /** {@inheritDoc} */
    @Override
    public FloatWrapper sub(FloatWrapper other) {
        return createFloatWrapper(floatValue() - other.floatValue());
    }

    /** {@inheritDoc} */
    @Override
    public FloatWrapper mul(FloatWrapper other) {
        return createFloatWrapper(floatValue() * other.floatValue());
    }

    /** {@inheritDoc} */
    @Override
    public FloatWrapper neg() {
        return createFloatWrapper(-floatValue());
    }

    /**
     * Creates numerical result data without changing either operand.
     * @param value the calculated scalar
     * @return a new wrapper of the scalar
     */
    protected FloatWrapper createFloatWrapper(float value) {
        return new FloatWrapperImpl(value);
    }
}
