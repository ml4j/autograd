package org.ml4j.autograd.arithmetic;

/**
 * Defines non-mutating numerical addition and multiplication.
 *
 * <p>A scalar wrapper can add or multiply its stored floats. A tensor type can
 * supply corresponding operations on compatible numerical data and shapes.
 *
 * <p>This capability is usable by plain numerical data and differentiable values.
 * Methods return non-null results without changing either operand. Numerical
 * implementations define compatibility and the precise meaning of multiplication;
 * the differentiable defaults require the ordinary product rule to be applicable.
 *
 * @param <V> the operand and result type
 */
public interface ArithmeticOperations<V extends ArithmeticOperations<?>> {
    /**
     * Returns the sum, for example 2 plus 3 equals 5 for scalars.
     * @param other a non-null compatible right operand
     * @return a non-null sum without mutating either operand
     */
    V add(V other);

    /**
     * Returns the product, for example 2 times 3 equals 6 for scalars.
     * @param other a non-null compatible right operand
     * @return a non-null product without mutating either operand
     */
    V mul(V other);
}
