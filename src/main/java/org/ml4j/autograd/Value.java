package org.ml4j.autograd;

import java.util.function.Supplier;

/**
 * Provides access to numerical data and the context needed to interpret it.
 *
 * <p>A value may represent a scalar, matrix, or tensor. Its numerical contents and
 * context remain stable throughout its lifetime, giving later computations a
 * consistent input. Stability does not require the same data or context object
 * identity; implementations and callers must not mutate exposed objects in ways
 * that change the represented value.
 *
 * <p>Data is exposed through a supplier to permit lazy evaluation. Implementations
 * decide whether to cache results. Gradients, computation graphs, naming, and data
 * replacement are separate responsibilities.
 *
 * @param <D> the numerical data type, for example {@link Float} or a tensor
 * @param <C> the context type, for example a shape, or {@link Void} when none is needed
 */
public interface Value<D, C> {

    /**
     * Returns a supplier that provides this value's non-null numerical data.
     *
     * <p>For a scalar, the supplier may return a {@link Float}. For a tensor,
     * {@code D} may be a tensor implementation type, so this method returns a
     * {@code Supplier<D>} that provides the tensor and its numerical contents.
     * This interface does not prescribe a tensor library or storage representation.
     *
     * <p>Obtaining the supplier does not evaluate the data. Repeated evaluations
     * represent the same numerical contents, whether computed afresh or cached.
     * Supplier and result object identities are not guaranteed. Evaluation failures
     * may propagate from the supplier; absence is not represented by a null result.
     *
     * @return a non-null data supplier whose successful evaluations return non-null data
     */
    Supplier<D> data();

    /**
     * Returns stable metadata describing how to interpret the numerical data.
     *
     * <p>For a tensor, {@code C} may be a {@code Shape} or {@code Size} type
     * describing its dimensions, for example {@code [2, 3]} for a matrix with
     * two rows and three columns. These are illustrative context types, not
     * dependencies required by this interface.
     *
     * <p>The context must agree with the data. Obtaining it does not evaluate the
     * data. Values needing no additional metadata may use {@link Void} and return null.
     *
     * @return the context, or null only when no additional metadata is needed
     */
    C context();
}
