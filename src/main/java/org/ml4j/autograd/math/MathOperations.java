package org.ml4j.autograd.math;

/**
 * Supplies non-mutating mathematical functions for scalars and tensors.
 *
 * <p>For example, exp(0) is 1 and log(1) is 0. Tensors apply each function
 * elementwise and preserve shape. Results are non-null and operands are unchanged.
 * Numerical implementations define precision, domains and non-finite behaviour;
 * no clipping or common cross-backend boundary policy is imposed.
 *
 * @param <V> the concrete numerical result type
 */
public interface MathOperations<V extends MathOperations<V>> {
    /**
     * Returns the exponential of each value, preserving tensor shape.
     * @return a non-null exponential result
     */
    V exp();

    /**
     * Returns the natural logarithm of each value, preserving tensor shape.
     * Zero, negative and non-finite inputs follow the numerical implementation.
     * @return a non-null logarithm result
     */
    V log();

    /**
     * Raises each value to a fixed float power, preserving tensor shape.
     * For example, [2, 3] raised to 2 becomes [4, 9]. Domain and signed-zero
     * behaviour follow the numerical implementation.
     * @param exponent the fixed numerical exponent
     * @return a non-null power result
     */
    V pow(float exponent);
}
