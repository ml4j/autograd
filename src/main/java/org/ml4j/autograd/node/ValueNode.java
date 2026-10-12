package org.ml4j.autograd.node;

import org.ml4j.autograd.Value;

import java.util.List;
import java.util.function.Consumer;

/**
 * Describes a value's place in a computation graph and its local backward rule.
 *
 * <p>A scalar or tensor supplied directly is a leaf with no inputs. An addition
 * result has the two operand nodes as inputs. Using the same operand twice records
 * that node twice, preserving both operand positions.
 *
 * <p>The backward callback receives the represented forward value. Gradient lookup
 * and contribution calculation belong to that callback; graph traversal belongs
 * to a separate coordinator. This interface requires only {@link Value}, without
 * assuming a particular gradient representation.
 *
 * @param <V> the represented value type
 */
public interface ValueNode<V extends Value<?, ?>> {

    /**
     * Returns the value represented by this node.
     *
     * <p>This may be a scalar value or a tensor value with shape context. Reading
     * the reference does not evaluate its numerical data.
     *
     * @return a non-null represented value
     */
    V value();

    /**
     * Returns the immediate inputs used to produce this value.
     *
     * <p>For scalar or tensor addition, the left operand precedes the right.
     * Repeated operands occupy separate positions even when they reference the
     * same node. A leaf returns an empty list.
     *
     * <p>The list is unmodifiable, non-null, and contains no null nodes. Inputs
     * may have different data or context types, for example in a conversion.
     *
     * @return the input nodes in operand order
     */
    List<? extends ValueNode<?>> inputs();

    /**
     * Installs the local backward callback, or removes it when supplied null.
     *
     * <p>For addition, the callback receives the sum value, not the sum's gradient.
     * A differentiable callback can read gradient state from that result and
     * contribute to the scalar or tensor operands.
     *
     * <p>Installing a callback replaces the previous one without invoking either.
     * This node does not extract gradients on the callback's behalf.
     *
     * @param resultValueBackwardFunction the callback consuming the forward value,
     *                                   or null to remove it
     */
    void setBackwardFunction(Consumer<V> resultValueBackwardFunction);

    /**
     * Returns the currently installed local backward callback.
     *
     * <p>Reading the callback does not invoke it or evaluate numerical data.
     *
     * @return the callback consuming the represented value, or null when none is installed
     */
    Consumer<V> backwardFunction();

    /**
     * Runs one local backward step for this node.
     *
     * <p>An addition node invokes its rule with the sum value. A node with no
     * installed callback does nothing.
     *
     * <p>Invokes the installed callback once with {@link #value()}. This method
     * does not traverse inputs, seed gradients, or extract a gradient. A graph
     * coordinator must ensure incoming contributions are ready before calling it.
     * Callback exceptions propagate to the caller; no rollback is guaranteed.
     */
    void backward();
}
