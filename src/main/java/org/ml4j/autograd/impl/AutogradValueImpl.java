package org.ml4j.autograd.impl;

import org.ml4j.autograd.AutogradValue;
import org.ml4j.autograd.Accumulatable;
import org.ml4j.autograd.Value;
import org.ml4j.autograd.node.GradNode;
import org.ml4j.autograd.node.ValueNode;
import org.ml4j.autograd.node.impl.GradNodeImpl;
import org.ml4j.autograd.node.impl.ValueNodeImpl;
import org.ml4j.autograd.operators.BinaryGradient;

import java.util.List;
import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.Map;
import java.util.Collections;
import java.util.Objects;
import java.util.function.BinaryOperator;
import java.util.function.Supplier;

/**
 * Provides shared node ownership and forward-operation construction for autograd values.
 *
 * <p>A float implementation supplies result construction and operation definitions,
 * either directly or through differentiable arithmetic interface defaults.
 * A future tensor implementation can supply tensor arithmetic and shape mapping
 * while using the same ordered graph inputs and gradient-state handles.
 *
 * <p>Each operation supplies its local derivative rules. Graph-wide passes order reachable
 * nodes by dependency, propagate new contributions, and retain previous leaf gradients.
 * Numerical subclasses define implicit seed policies; explicit seeds work generically.
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
     * Constructs a lazy binary result and installs its local backward callback.
     *
     * <p>The context mapper runs once at construction; numerical evaluation is deferred.
     * Scalar operands may have no context; tensor implementations map compatible shapes.
     * The node callback receives the forward result value and extracts its gradient
     * before invoking the separate left/right derivative rules.
     *
     * @param other the right operand
     * @param forward the numerical operation
     * @param leftGradient the left operand's contribution rule
     * @param rightGradient the right operand's contribution rule
     * @param contextMapper maps operand metadata to output metadata
     * @return the concrete result, requiring gradients if either operand does
     */
    public final V applyBinaryOperator(V other, BinaryOperator<D> forward,
                                          BinaryGradient<V> leftGradient, BinaryGradient<V> rightGradient,
                                          BinaryOperator<C> contextMapper) {
        Objects.requireNonNull(other, "other");
        Objects.requireNonNull(forward, "forward");
        Objects.requireNonNull(leftGradient, "leftGradient");
        Objects.requireNonNull(rightGradient, "rightGradient");
        Objects.requireNonNull(contextMapper, "contextMapper");
        C outputContext = contextMapper.apply(context(), other.context());
        V result = createAutogradValue(() -> forward.apply(data().get(), other.data().get()),
                outputContext, requiresGrad() || other.requiresGrad(), List.of(valueNode(), other.valueNode()));
        result.valueNode().setBackwardFunction(resultValue -> {
            if (!resultValue.requiresGrad()) {
                return;
            }
            resultValue.gradNode().gradient().ifPresent(incomingGradient -> {
                if (requiresGrad()) {
                    gradNode().accumulate(leftGradient.apply(incomingGradient, self(), other));
                }
                if (other.requiresGrad()) {
                    // Duplicate operands still have two derivative rules and two contributions.
                    other.gradNode().accumulate(rightGradient.apply(incomingGradient, self(), other));
                }
            });
        });
        return result;
    }

    /**
     * Maps operand metadata to the result context, such as tensor shape or null for scalars.
     * @param left left context
     * @param right right context
     * @return compatible output context
     */
    public abstract C mapContext(C left, C right);

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

    /** {@inheritDoc} */
    @Override
    public void backward(V gradient) {
        Objects.requireNonNull(gradient, "seed gradient");
        if (!requiresGrad()) {
            throw new IllegalStateException("The backward root must require gradients");
        }
        // Discover and validate the original graph before derivative operations add new nodes.
        List<ValueNode<?>> order = new ArrayList<>();
        visit(valueNode(), new IdentityHashMap<>(), order);
        List<Runnable> restoreLeaves = new ArrayList<>();
        for (ValueNode<?> node : order) {
            AutogradValue<?, ?, ?> owner = (AutogradValue<?, ?, ?>) node.value();
            if (node.inputs().isEmpty()) {
                saveLeafGradient(owner.gradNode(), restoreLeaves);
            }
            owner.gradNode().reset();
        }
        gradNode().accumulate(gradient);
        Collections.reverse(order);
        for (ValueNode<?> node : order) {
            AutogradValue<?, ?, ?> owner = (AutogradValue<?, ?, ?>) node.value();
            if (owner.requiresGrad() && owner.gradNode().gradient().isPresent()) {
                node.backward();
            }
        }
        // Restore previous leaves AFTER propagation, preventing stale contributions from running again.
        restoreLeaves.forEach(Runnable::run);
    }

    /** Retains typed gradient references for restoration without mutating earlier expressions. */
    private static <G extends Value<?, ?> & Accumulatable<G>> void saveLeafGradient(
            GradNode<G> node, List<Runnable> restoreLeaves) {
        node.gradient().ifPresent(previous -> restoreLeaves.add(() -> node.accumulate(previous)));
    }

    /** Produces dependency-first order by identity; a false entry marks a node still being visited. */
    private static void visit(ValueNode<?> node, Map<ValueNode<?>, Boolean> visited, List<ValueNode<?>> order) {
        Boolean complete = visited.get(node);
        if (complete != null) {
            if (!complete) throw new IllegalStateException("Cycle in computation graph");
            return;
        }
        if (!(node.value() instanceof AutogradValue<?, ?, ?>)) {
            throw new IllegalArgumentException("Backward graph inputs must represent autograd values");
        }
        visited.put(node, false);
        for (ValueNode<?> input : node.inputs()) {
            visit(input, visited, order);
        }
        visited.put(node, true);
        order.add(node);
    }

    /** {@inheritDoc} */
    @Override
    public void backward() {
        if (!requiresGrad()) {
            throw new IllegalStateException("The backward root must require gradients");
        }
        backward(defaultGradient());
    }

    /**
     * Supplies this numerical type's implicit seed, if defined.
     *
     * <p>Float scalars use one. Tensor subclasses can define a compatible seed
     * policy or keep the default requirement for an explicit seed.
     *
     * @return the implicit seed
     * @throws UnsupportedOperationException unless the subclass defines an implicit seed
     */
    protected V defaultGradient() {
        throw new UnsupportedOperationException("Supply an explicit seed gradient");
    }
}
