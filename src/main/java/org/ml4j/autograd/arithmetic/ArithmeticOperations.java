package org.ml4j.autograd.arithmetic;

/**
 * Defines non-mutating numerical arithmetic and proposed data-access and comparison capabilities.
 *
 * <p>A scalar wrapper can add or multiply its stored floats. A tensor type can
 * supply corresponding operations on compatible numerical data and shapes.
 *
 * <p>This capability is usable by plain numerical data and differentiable values.
 * Methods return non-null results without changing either operand. Numerical
 * implementations define compatibility and the precise meaning of multiplication;
 * the differentiable defaults require the ordinary product rule to be applicable.
 *
 * <p>Scalar overloads, float-array access, and threshold comparisons are provisional
 * integration signatures. Their defaults throw UnsupportedOperationException until
 * implemented; conversion, broadcasting, and comparison policies remain unresolved.
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
     * Proposes reading numerical contents as a float array.
     *
     * <p>Array ownership, tensor flattening order, and conversion rules remain unresolved.
     * This default does not evaluate data.
     *
     * @return numerical contents when implemented
     * @throws UnsupportedOperationException always in this unimplemented default
     */
    default float[] dataAsFloatArray() {
        throw new UnsupportedOperationException("dataAsFloatArray is not implemented");
    }

    /**
     * Proposes adding a scalar constant without mutating this value.
     *
     * <p>Constant construction and tensor broadcasting/context rules remain unresolved.
     *
     * @param other the scalar constant
     * @return the arithmetic result when implemented
     * @throws UnsupportedOperationException always in this unimplemented default
     */
    default V add(float other) {
        throw new UnsupportedOperationException("add(float) is not implemented");
    }

    /**
     * Proposes subtracting a scalar constant without mutating this value.
     *
     * <p>Constant construction and tensor broadcasting/context rules remain unresolved.
     *
     * @param other the scalar constant
     * @return the arithmetic result when implemented
     * @throws UnsupportedOperationException always in this unimplemented default
     */
    default V sub(float other) {
        throw new UnsupportedOperationException("sub(float) is not implemented");
    }

    /**
     * Proposes multiplying by a scalar constant without mutating this value.
     *
     * <p>Constant construction and tensor broadcasting/context rules remain unresolved.
     *
     * @param other the scalar constant
     * @return the arithmetic result when implemented
     * @throws UnsupportedOperationException always in this unimplemented default
     */
    default V mul(float other) {
        throw new UnsupportedOperationException("mul(float) is not implemented");
    }

    /**
     * Proposes dividing by a scalar constant without mutating this value.
     *
     * <p>Constant construction and tensor broadcasting/context rules remain unresolved.
     *
     * @param other the scalar constant
     * @return the arithmetic result when implemented
     * @throws UnsupportedOperationException always in this unimplemented default
     */
    default V div(float other) {
        throw new UnsupportedOperationException("div(float) is not implemented");
    }

    /**
     * Proposes comparing this value with a scalar threshold: greater than.
     *
     * <p>Result encoding, tensor elementwise rules, non-finite behaviour, and
     * differentiation policy remain unresolved. Returning V promises no derivative rule.
     *
     * @param value the scalar threshold
     * @return the comparison result when implemented
     * @throws UnsupportedOperationException always in this unimplemented default
     */
    default V gt(float value) {
        throw new UnsupportedOperationException("gt(float) is not implemented");
    }

    /**
     * Proposes comparing this value with a scalar threshold: greater than or equal to.
     *
     * <p>Result encoding, tensor elementwise rules, non-finite behaviour, and
     * differentiation policy remain unresolved. Returning V promises no derivative rule.
     *
     * @param value the scalar threshold
     * @return the comparison result when implemented
     * @throws UnsupportedOperationException always in this unimplemented default
     */
    default V gte(float value) {
        throw new UnsupportedOperationException("gte(float) is not implemented");
    }
}
