package org.ml4j.autograd.node.impl;

import org.ml4j.autograd.Value;
import org.ml4j.autograd.node.ValueNode;

import java.util.List;
import java.util.Objects;
import java.util.function.Consumer;
import java.util.function.Supplier;

/**
 * Holds a forward value's input nodes and local backward callback.
 *
 * <p>A scalar leaf uses an empty input list. A scalar or tensor operation result
 * supplies its ordered operand nodes, including duplicate references when an
 * operand is used more than once.
 *
 * <p>The represented value is obtained from a supplier only when requested. This
 * permits a value to create its node before construction finishes. Inputs are
 * copied immutably; callback replacement changes only the installed rule.
 * Instances are intended for sequential use and are not thread-safe.
 *
 * @param <V> the represented value type
 */
public final class ValueNodeImpl<V extends Value<?, ?>> implements ValueNode<V> {

    private final Supplier<V> valueSupplier;
    private final List<? extends ValueNode<?>> inputs;
    private Consumer<V> resultValueBackwardFunction;

    /**
     * Creates a leaf node with no inputs or installed callback.
     *
     * @param valueSupplier a supplier of the non-null represented value; not invoked here
     * @throws NullPointerException if the supplier is null
     */
    public ValueNodeImpl(Supplier<V> valueSupplier) {
        this(valueSupplier, List.of());
    }

    /**
     * Creates a node with a deferred value reference and ordered inputs.
     *
     * <p>For addition, pass the left and right operand nodes. A tensor conversion
     * can supply an input whose data or context type differs from the result's.
     *
     * <p>The input list is copied without deduplication. The value supplier is not
     * invoked at construction and must consistently represent this node's value.
     * No callback is initially installed.
     *
     * @param valueSupplier a supplier of the non-null represented value
     * @param inputs the immediate input nodes in operand order
     * @throws NullPointerException if the supplier, input list, or any input node is null
     */
    public ValueNodeImpl(Supplier<V> valueSupplier, List<? extends ValueNode<?>> inputs) {
        this.valueSupplier = Objects.requireNonNull(valueSupplier, "valueSupplier");
        this.inputs = List.copyOf(inputs);
    }

    /**
     * {@inheritDoc}
     *
     * @throws NullPointerException if the supplier returns null
     */
    @Override
    public V value() {
        return Objects.requireNonNull(valueSupplier.get(), "represented value");
    }

    /** {@inheritDoc} */
    @Override
    public List<? extends ValueNode<?>> inputs() {
        return inputs;
    }

    /** {@inheritDoc} */
    @Override
    public void setBackwardFunction(Consumer<V> resultValueBackwardFunction) {
        this.resultValueBackwardFunction = resultValueBackwardFunction;
    }

    /** {@inheritDoc} */
    @Override
    public Consumer<V> backwardFunction() {
        return resultValueBackwardFunction;
    }

    /** {@inheritDoc} */
    @Override
    public void backward() {
        if (resultValueBackwardFunction != null) {
            // Supply the forward/result value. The callback handles gradient lookup.
            resultValueBackwardFunction.accept(value());
        }
    }
}
