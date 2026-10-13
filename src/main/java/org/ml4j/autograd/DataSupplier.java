package org.ml4j.autograd;

import java.util.function.Supplier;

/**
 * Provides data access without requiring context, arithmetic, or autograd state.
 *
 * <p>A scalar implementation can supply a Float; a tensor adapter can supply its
 * numerical tensor type. Data evaluation is deferred until the supplier is invoked.
 * Implementations define caching, data stability, and whether supplied data may be null.
 * Value adds stronger guarantees for stable, non-null numerical contents.
 *
 * @param <D> the supplied data type
 */
public interface DataSupplier<D> {

    /**
     * Returns a supplier for this object's data without evaluating it.
     *
     * <p>Supplier and result identity and caching are implementation-defined.
     * Evaluation failures may propagate from the supplier.
     *
     * @return a non-null data supplier
     */
    Supplier<D> data();
}
