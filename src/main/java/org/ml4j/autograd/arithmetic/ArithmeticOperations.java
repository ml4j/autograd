package org.ml4j.autograd.arithmetic;

/**
 * Defines non-mutating numerical arithmetic, data access, and comparisons.
 *
 * <p>A scalar wrapper can add or multiply its stored floats. A tensor type can
 * supply corresponding operations on compatible numerical data and shapes.
 *
 * <p>This capability is usable by plain numerical data and differentiable values.
 * Methods return non-null results without changing either operand. Numerical
 * implementations define compatibility and the precise meaning of multiplication;
 * the differentiable defaults require the ordinary product rule to be applicable.
 *
 * <p>This interface declares capabilities without supplying implementations.
 * Numerical backends implement them; differentiable implementations can reuse
 * the defaults in DifferentiableArithmeticOperations.
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
     * Returns the quotient, for example 6 divided by 2 equals 3.
     *
     * <p>Tensor types define compatible elementwise division and shape rules.
     * Numerical implementations define zero and non-finite behaviour; float types
     * use IEEE arithmetic. Neither operand is mutated.
     *
     * @param other a non-null compatible denominator
     * @return a non-null quotient
     */
    V div(V other);

    /**
     * Returns the arithmetic negation, for example 2 becomes -2.
     *
     * <p>Tensor negation changes each element's sign while preserving shape.
     * The operand's numerical data, graph, and gradient state are unchanged.
     *
     * @return a non-null negated value
     */
    V neg();

    /**
     * Reads numerical contents as a float array.
     *
     * <p>Storage may be copied or shared and must be treated as read-only. Backend
     * implementations document flattening/conversion rules. Access may evaluate data.
     *
     * @return non-null numerical contents
     */
    float[] dataAsFloatArray();

    /**
     * Returns the result of adding a scalar constant without mutating this value.
     *
     * <p>The scalar is a fixed constant. Numerical implementations define
     * compatible elementwise tensor behaviour and numerical boundary rules.
     *
     * @param other the scalar constant
     * @return the non-null arithmetic result
     */
    V add(float other);

    /**
     * Returns the result of subtracting a scalar constant without mutating this value.
     *
     * <p>The scalar is a fixed constant. Numerical implementations define
     * compatible elementwise tensor behaviour and numerical boundary rules.
     *
     * @param other the scalar constant
     * @return the non-null arithmetic result
     */
    V sub(float other);

    /**
     * Returns the result of multiplying by a scalar constant without mutating this value.
     *
     * <p>The scalar is a fixed constant. Numerical implementations define
     * compatible elementwise tensor behaviour and numerical boundary rules.
     *
     * @param other the scalar constant
     * @return the non-null arithmetic result
     */
    V mul(float other);

    /**
     * Returns the result of dividing by a scalar constant without mutating this value.
     *
     * <p>The scalar is a fixed constant. Numerical implementations define
     * compatible elementwise tensor behaviour and numerical boundary rules.
     *
     * @param other the scalar constant
     * @return the non-null arithmetic result
     */
    V div(float other);

    /**
     * Compares this value with a scalar threshold: greater than.
     *
     * <p>Returns a float mask: one where true, zero where false. Numerical backends
     * define non-finite behaviour. Differentiable comparisons return constant masks.
     *
     * @param value the scalar threshold
     * @return the non-null comparison mask
     */
    V gt(float value);

    /**
     * Compares this value with a scalar threshold: greater than or equal to.
     *
     * <p>Returns a float mask: one where true, zero where false. Numerical backends
     * define non-finite behaviour. Differentiable comparisons return constant masks.
     *
     * @param value the scalar threshold
     * @return the non-null comparison mask
     */
    V gte(float value);
}
