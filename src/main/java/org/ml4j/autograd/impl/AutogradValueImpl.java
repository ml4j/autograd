package org.ml4j.autograd.impl;

import org.ml4j.autograd.AutogradValue;
import org.ml4j.autograd.node.GradNode;
import org.ml4j.autograd.node.ValueNode;
import org.ml4j.autograd.node.impl.GradNodeImpl;
import org.ml4j.autograd.node.impl.ValueNodeImpl;

import java.util.List;
import java.util.Objects;
import java.util.function.BinaryOperator;
import java.util.function.Supplier;

/**
 * Provides shared node ownership and forward-operation construction for autograd values.
 *
 * <p>A float implementation supplies numerical addition and result construction.
 * A future tensor implementation can supply tensor arithmetic and shape mapping
 * while using the same ordered graph inputs and gradient-state handles.
 *
 * <p>Addition has a local backward rule, while graph-wide backward entry points
 * remain unimplemented. This partial implementation does not yet fulfil the complete
 * backward contract of {@link AutogradValue}.
 * Supplied data and context must satisfy {@link org.ml4j.autograd.Value}'s stability
 * contract. The base class adds no caching and is intended for sequential use.
 *
 * @param <V> the concrete autograd value type
 * @param <D> the numerical data type
 * @param <C> the context type, such as shape or Void
 */
public abstract class AutogradValueImpl<V extends AutogradValue<V, D, C>, D, C>
        implements AutogradValue<V, D, C> {

    private final Supplier<D> data;
    private final C context;
    private final boolean requiresGrad;
    private final ValueNode<V> valueNode;
    private final GradNode<V> gradNode;

    /**
     * Creates the shared state without evaluating data or calling subclass hooks.
     *
     * @param data a supplier of stable, non-null numerical data
     * @param context stable metadata, or null when none is needed
     * @param requiresGrad whether differentiation should calculate a gradient
     * @param inputs ordered operands, empty for a leaf
     * @throws NullPointerException if data, inputs, or an input node is null
     */
    protected AutogradValueImpl(Supplier<D> data, C context, boolean requiresGrad,
                                List<? extends ValueNode<?>> inputs) {
        Supplier<D> source = Objects.requireNonNull(data, "data");
        this.data = () -> Objects.requireNonNull(source.get(), "numerical data");
        this.context = context;
        this.requiresGrad = requiresGrad;
        // The supplier defers the subclass hook until the node's represented value is read.
        this.valueNode = new ValueNodeImpl<>(this::self, inputs);
        this.gradNode = new GradNodeImpl<>();
    }

    /**
     * Supplies the concrete owner when the forward node is read.
     *
     * <p>A scalar or tensor subclass normally returns this. The constructor does
     * not invoke this hook; the reference must consistently represent this instance.
     *
     * @return this value as its concrete type, never null
     */
    protected abstract V self();

    /** {@inheritDoc} */
    @Override public final Supplier<D> data() { return data; }

    /** {@inheritDoc} */
    @Override public final C context() { return context; }

    /** {@inheritDoc} */
    @Override public final boolean requiresGrad() { return requiresGrad; }

    /** {@inheritDoc} */
    @Override public final ValueNode<V> valueNode() { return valueNode; }

    /** {@inheritDoc} */
    @Override public final GradNode<V> gradNode() { return gradNode; }

    /**
     * Builds a lazy addition result with both operands recorded as inputs.
     *
     * <p>Scalar arithmetic or tensor addition is supplied by the concrete type.
     * Forward construction leaves operand data and gradient state unchanged.
     * Each tracked operand receives the result's incoming gradient unchanged when
     * the result node's local backward callback is invoked.
     *
     * @param other the right operand
     * @return the new forward sum
     * @throws NullPointerException if other is null
     */
    @Override
    public V add(V other) {
        V result = applyBinaryOperator(other, this::addData, this::mapContext);
        result.valueNode().setBackwardFunction(resultValue -> {
            if (!resultValue.requiresGrad()) {
                return;
            }
            resultValue.gradNode().gradient().ifPresent(incomingGradient -> {
                // Addition has unit partial derivatives: each tracked operand receives the seed.
                if (requiresGrad()) {
                    gradNode().accumulate(incomingGradient);
                }
                if (other.requiresGrad()) {
                    // Do not deduplicate: x.add(x) contributes once for each operand position.
                    other.gradNode().accumulate(incomingGradient);
                }
            });
        });
        return result;
    }

    /**
     * Constructs a forward binary result with mapped context and ordered dependencies.
     *
     * <p>The context mapper runs once at construction; numerical evaluation is deferred.
     * Scalar operands may have no context; tensor implementations map compatible shapes.
     *
     * @param other the right operand
     * @param forward the numerical operation
     * @param contextMapper maps operand metadata to output metadata
     * @return the concrete result, requiring gradients if either operand does
     */
    protected final V applyBinaryOperator(V other, BinaryOperator<D> forward, BinaryOperator<C> contextMapper) {
        Objects.requireNonNull(other, "other");
        Objects.requireNonNull(forward, "forward");
        Objects.requireNonNull(contextMapper, "contextMapper");
        C outputContext = contextMapper.apply(context(), other.context());
        V result = createAutogradValue(() -> forward.apply(data().get(), other.data().get()),
                outputContext, requiresGrad() || other.requiresGrad(), List.of(valueNode(), other.valueNode()));
        // The concrete operation installs its local derivative rule after forward construction.
        return result;
    }

    /**
     * Adds numerical data without mutating either operand.
     * @param left left scalar or tensor data
     * @param right right scalar or tensor data
     * @return the numerical sum
     */
    protected abstract D addData(D left, D right);

    /**
     * Maps operand metadata to the result context, such as tensor shape or null for scalars.
     * @param left left context
     * @param right right context
     * @return compatible output context
     */
    protected abstract C mapContext(C left, C right);

    /**
     * Constructs the concrete operation result, preserving the supplied metadata and inputs.
     * @param dataSupplier lazy result data
     * @param context mapped output metadata
     * @param requiresGrad combined operand gradient requirement
     * @param inputs ordered dependencies, preserving duplicate operands
     * @return the new result value
     */
    protected abstract V createAutogradValue(Supplier<D> dataSupplier, C context, boolean requiresGrad,
                                             List<? extends ValueNode<?>> inputs);

    /**
     * Placeholder for graph-wide backward execution with an explicit seed.
     * @param gradient the intended incoming seed
     * @throws UnsupportedOperationException always in this forward-only increment
     */
    @Override public void backward(V gradient) {
        throw new UnsupportedOperationException("Backward passes are not implemented yet");
    }

    /**
     * Placeholder for graph-wide backward execution with an implicit seed.
     * @throws UnsupportedOperationException always in this forward-only increment
     */
    @Override public void backward() {
        throw new UnsupportedOperationException("Backward passes are not implemented yet");
    }
}
