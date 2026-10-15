package org.ml4j.autograd.scalar.impl;

import org.ml4j.autograd.scalar.FloatWrapper;

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
     * Returns a sigmoid computed with double intermediates and rounded to float.
     *
     * <p>Both signed zeros yield 0.5; negative infinity yields zero, positive infinity
     * yields one, and NaN remains NaN. Finite tails may round to zero or one.
     *
     * @return a new non-null numerical wrapper
     */
    @Override
    public FloatWrapper sigmoid() {
        double exponential = Math.exp(-Math.abs((double) value));
        return createFloatWrapper((float) (value >= 0f
                ? 1d / (1d + exponential) : exponential / (1d + exponential)));
    }

    /** Returns Java Math.exp rounded to float, including overflow and underflow. */
    @Override
    public FloatWrapper exp() {
        return createFloatWrapper((float) Math.exp(value));
    }

    /** Returns Java Math.log rounded to float, including NaN and infinity at domain boundaries. */
    @Override
    public FloatWrapper log() {
        return createFloatWrapper((float) Math.log(value));
    }

    /** Returns Java Math.pow rounded to float, preserving its domain and signed-zero rules. */
    @Override
    public FloatWrapper pow(float exponent) {
        return createFloatWrapper((float) Math.pow(value, exponent));
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
