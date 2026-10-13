package org.ml4j.autograd;

import org.ml4j.autograd.node.GradNode;
import org.ml4j.autograd.node.ValueNode;

/**
 * Connects numerical data to its computation node and accumulated gradient state.
 *
 * <p>A scalar addition result or tensor operation result has a value node describing
 * its inputs. Its gradient node holds contributions produced by differentiation.
 * A gradient can itself be an autograd value with its own computation graph,
 * allowing gradient expressions to be differentiated for higher derivatives.
 * For example, the gradient of a scalar cube can be differentiated again; tensor
 * operation rules can support Hessian-vector products in the same way.
 *
 * <p>Backward currently records differentiable gradient computations (broadly
 * equivalent to {@code createGraph = true}) and preserves the original graph for
 * reuse (the purpose of {@code keepGraph}, also called {@code retainGraph}). There
 * are no configuration flags for either policy. Constant derivatives need not
 * require gradients; higher-derivative support depends on the operation rules.
 *
 * <p>These policies may be revisited for first-order training without derivative
 * graphs, tensor memory pressure, explicit release of saved backward buffers, or
 * operations with limited higher-derivative support. Such configuration requires
 * a separately agreed contract; it is not provided by this interface today.
 *
 * <p>This value coordinates graph-wide backward passes. Value nodes execute local
 * rules; gradient nodes store contributions. Addition follows the non-mutating
 * {@link Accumulatable} contract, retaining dependencies on the result.
 *
 * @param <V> the concrete autograd value type, also used to represent gradients
 * @param <D> the numerical data type, such as Float or a tensor implementation
 * @param <C> the context type, such as shape/size or Void when no metadata is needed
 */
public interface AutogradValue<V extends AutogradValue<V, D, C>, D, C>
        extends Value<D, C>, Accumulatable<V> {

    /**
     * Returns the node describing this value's computation and inputs.
     *
     * <p>A directly supplied scalar or tensor is a leaf with no inputs. An operation
     * result records its operands as input nodes.
     *
     * @return a non-null node representing this value
     */
    ValueNode<V> valueNode();

    /**
     * Returns a stable handle to this value's gradient state.
     *
     * <p>Retain the node and read {@code gradNode().gradient()} to obtain the current
     * scalar or tensor gradient. That gradient may be absent or replaced later.
     *
     * <p>The same node instance is returned throughout this value's lifetime,
     * including after accumulation, backward passes, and reset. Previously returned
     * gradient expressions are not refreshed or mutated when the held reference changes.
     *
     * @return the non-null gradient node owned by this value
     */
    GradNode<V> gradNode();

    /**
     * Indicates whether differentiation should calculate a gradient for this value.
     *
     * <p>A constant scalar or tensor may have a computation node while returning
     * false. Returning true requests gradient calculation; it does not imply that
     * a gradient has already been calculated.
     *
     * @return whether this value requires a gradient
     */
    boolean requiresGrad();

    /**
     * Changes whether future graphs calculate this leaf's gradient.
     *
     * <p>For example, a scalar or tensor parameter can enable tracking before
     * building a training graph and disable it before building an inference graph.
     * Only leaves (values with no graph inputs) can change this flag. Set it before
     * constructing the next graph; do not change it while a graph involving this
     * leaf is still needed for backward or higher derivatives.
     *
     * <p>Existing operation results keep their original gradient requirements.
     * Numerical data, context, graph nodes, and stored gradients are unchanged;
     * clearing a gradient requires an explicit gradient-node reset. The implementation
     * does not detect outstanding graphs or support concurrent toggling/backward.
     *
     * @param requiresGrad whether future differentiation should calculate this leaf's gradient
     * @throws IllegalStateException if this value has graph inputs, even if the flag is unchanged
     */
    void setRequiresGrad(boolean requiresGrad);

    /**
     * Propagates a supplied seed gradient through this value's reachable graph.
     *
     * <p>A scalar seed of one differentiates the scalar result itself. For a tensor
     * result, a compatible tensor seed specifies the incoming gradient. Seed shape
     * and context compatibility follow the concrete numerical type's rules.
     *
     * <p>Shared nodes are visited once by node identity, after all downstream
     * contributions arrive. Duplicate operand positions still contribute separately.
     * Each pass propagates only its new contributions: gradients of leaves (nodes
     * with no inputs) accumulate across passes; intermediate gradients are replaced.
     * Constants receive no contributions from the pass. Reset leaf gradient nodes
     * explicitly to discard previous contributions.
     *
     * <p>Gradient computations retain differentiable dependencies for higher derivatives;
     * constant derivative expressions need not require gradients themselves. Gradient
     * value identity is not guaranteed, while gradient-node identity remains stable.
     * Existing gradient values and their graphs are not mutated by replacement.
     * This records derivative computations as with {@code createGraph = true};
     * no detached-gradient mode is provided.
     *
     * <p>The original graph remains available for another pass: backward does not
     * release its inputs, callbacks, or numerical suppliers. This serves the purpose
     * of {@code keepGraph}/{@code retainGraph}, independently of derivative recording.
     * Resetting a gradient node clears its held reference, not forward graph state
     * or previously returned expressions. Unreachable objects can still be garbage
     * collected. These policies do not change lazy forward evaluation.
     *
     * <p>Execution is sequential; concurrent passes over shared values are unsupported.
     * Callback failures may leave partially updated gradient state; no rollback is
     * guaranteed. Graph-wide coordination is distinct from {@link ValueNode#backward()}.
     *
     * @param gradient a non-null seed compatible with this value's numerical data and context
     * @throws NullPointerException if the seed is null
     * @throws IllegalStateException if this value does not require gradients or its graph contains a cycle
     */
    void backward(V gradient);

    /**
     * Runs a backward pass using this numerical type's implicit seed.
     *
     * <p>A float scalar uses one. Tensor types must define their own implicit seed
     * policy, or require callers to use the explicit-seed overload.
     *
     * <p>The derivative-recording, graph-retention, traversal, accumulation, and failure rules are the same as for
     * {@link #backward(AutogradValue)}.
     *
     * @throws UnsupportedOperationException if this type does not define an implicit seed
     * @throws IllegalStateException if this value does not require gradients or its graph contains a cycle
     */
    void backward();
}
