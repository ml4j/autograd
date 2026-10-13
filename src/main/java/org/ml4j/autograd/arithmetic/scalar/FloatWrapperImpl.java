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

    /**
     * Returns ordinary float division, preserving IEEE infinities and NaN.
     * @param other the non-null denominator
     * @return a new numerical quotient wrapper
     * @throws NullPointerException if other is null
     */
    @Override
    public FloatWrapper div(FloatWrapper other) {
        return createFloatWrapper(floatValue() / other.floatValue());
    }

    /** {@inheritDoc} */
    @Override
    public FloatWrapper neg() {
        return createFloatWrapper(-floatValue());
    }

    /** Returns an independent one-element float array containing this scalar. */
    @Override
    public float[] dataAsFloatArray() {
        return new float[]{value};
    }

    /** Returns ordinary float arithmetic with a fixed constant, including IEEE boundaries. */
    @Override
    public FloatWrapper add(float other) {
        return createFloatWrapper(value + other);
    }

    /** Returns ordinary float arithmetic with a fixed constant, including IEEE boundaries. */
    @Override
    public FloatWrapper sub(float other) {
        return createFloatWrapper(value - other);
    }

    /** Returns ordinary float arithmetic with a fixed constant, including IEEE boundaries. */
    @Override
    public FloatWrapper mul(float other) {
        return createFloatWrapper(value * other);
    }

    /** Returns ordinary float arithmetic with a fixed constant, including IEEE boundaries. */
    @Override
    public FloatWrapper div(float other) {
        return createFloatWrapper(value / other);
    }

    /** Returns a zero/one mask using Java float comparison; NaN comparisons are false. */
    @Override
    public FloatWrapper gt(float threshold) {
        return createFloatWrapper(value > threshold ? 1f : 0f);
    }

    /** Returns a zero/one mask using Java float comparison; NaN comparisons are false. */
    @Override
    public FloatWrapper gte(float threshold) {
        return createFloatWrapper(value >= threshold ? 1f : 0f);
    }

    /**
     * Returns max(0, value) using Java float arithmetic.
     *
     * <p>Both signed zeros produce positive zero; NaN remains NaN, positive infinity
     * remains positive infinity, and negative infinity becomes zero.
     *
     * @return a new non-null numerical wrapper
     */
    @Override
    public FloatWrapper relu() {
        return createFloatWrapper(Math.max(0f, value));
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
