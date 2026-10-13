package org.ml4j.autograd.arithmetic;

/**
 * Defines non-mutating numerical addition, subtraction, multiplication, and negation.
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
     * Returns the difference, for example 5 minus 3 equals 2.
     *
     * <p>Tensors subtract compatible elements according to their numerical shape rules.
     * Neither operand's data, computation graph, nor gradient state is changed.
     *
     * @param other a non-null compatible right operand
     * @return a non-null difference
     */
    V sub(V other);

    /**
     * Returns the product, for example 2 times 3 equals 6 for scalars.
     * @param other a non-null compatible right operand
     * @return a non-null product without mutating either operand
     */
    V mul(V other);

    /**
     * Returns the arithmetic negation, for example 2 becomes -2.
     *
     * <p>Tensor negation changes each element's sign while preserving shape.
     * The operand's numerical data, graph, and gradient state are unchanged.
     *
     * @return a non-null negated value
     */
    V neg();
}
