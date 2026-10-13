package org.ml4j.autograd.arithmetic.scalar;

import org.ml4j.autograd.arithmetic.ArithmeticOperations;

/**
 * Exposes a float with non-mutating arithmetic, without autograd graph state.
 *
 * <p>Unlike Java's Float, this type implements the library's arithmetic capability.
 * It acts as a scalar example of numerical data a tensor adapter could also provide.
 */
public interface FloatWrapper extends ArithmeticOperations<FloatWrapper> {
    /**
     * Returns the numerical scalar, for example 2.0 for a wrapper constructed from 2f.
     * @return the stored primitive float
     */
    float getFloat();
}
