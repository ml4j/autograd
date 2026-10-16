package org.ml4j.autograd.impl;

import org.junit.jupiter.api.Test;
import org.ml4j.autograd.node.ValueNode;

import java.util.List;
import java.util.function.Consumer;
import java.util.function.Supplier;

import static org.junit.jupiter.api.Assertions.*;

class AfterBackwardTest {
    @Test
    void rootHookSeesFinalRestoredTotalsAndOriginalDeduplicatedNodes() {
        var x = new TestValue(2, true);
        var constant = new TestValue(3, false);
        var duplicate = x.add(x);
        var root = duplicate.add(constant);
        x.gradNode().accumulate(new TestValue(5, false));
        root.callback = nodes -> {
            assertEquals(7f, x.gradNode().gradient().orElseThrow().data().get());
            assertEquals(List.of(root.valueNode(), constant.valueNode(), duplicate.valueNode(), x.valueNode()), nodes);
            assertThrows(UnsupportedOperationException.class, nodes::clear);
        };
        root.backward();
        assertEquals(1, root.hooks);
        assertEquals(0, x.hooks);
        assertEquals(0, duplicate.hooks);
        assertEquals(0, constant.hooks);
    }

    @Test
    void explicitSeedAndRepeatedPassesCallHookOnceEachWithoutEagerData() {
        var x = new TestValue(() -> { throw new AssertionError("No numerical reads needed"); }, true, List.of());
        var root = x.add(x);
        root.callback = nodes -> assertEquals(2, nodes.size());
        root.backward(new TestValue(3, false));
        assertEquals(1, root.hooks);
        root.backward(new TestValue(2, false));
        assertEquals(2, root.hooks);
        assertEquals(10f, x.gradNode().gradient().orElseThrow().data().get());
    }

    @Test
    void invalidOrFailedPassDoesNotInvokeHook() {
        var x = new TestValue(2, true);
        var root = x.add(x);
        assertThrows(NullPointerException.class, () -> root.backward(null));
        assertEquals(0, root.hooks);
        root.valueNode().setBackwardFunction(value -> { throw new IllegalStateException("propagation failed"); });
        assertThrows(IllegalStateException.class, root::backward);
        assertEquals(0, root.hooks);
        var frozen = new TestValue(2, false);
        assertThrows(IllegalStateException.class, frozen::backward);
        assertEquals(0, frozen.hooks);
    }

    @Test
    void hookFailurePropagatesAfterCompletedGradientsWithoutRollback() {
        var x = new TestValue(2, true);
        var root = x.add(x);
        var failure = new IllegalStateException("hook failed");
        root.callback = nodes -> { throw failure; };
        assertSame(failure, assertThrows(IllegalStateException.class, root::backward));
        assertEquals(1, root.hooks);
        assertEquals(2f, x.gradNode().gradient().orElseThrow().data().get());
    }

    private static class TestValue extends AutogradValueImpl<TestValue, Float, Void> {
        private int hooks;
        private Consumer<List<ValueNode<?>>> callback = nodes -> {};

        TestValue(float data, boolean tracked) { this(() -> data, tracked, List.of()); }
        TestValue(Supplier<Float> data, boolean tracked, List<? extends ValueNode<?>> inputs) {
            super(data, null, tracked, inputs);
        }
        @Override public TestValue add(TestValue other) {
            return applyBinaryOperator(other, Float::sum, (g, a, b) -> g, (g, a, b) -> g, (a, b) -> null);
        }
        @Override public Void mapContext(Void a, Void b) { return null; }
        @Override protected TestValue self() { return this; }
        @Override protected TestValue defaultGradient() { return new TestValue(1, false); }
        @Override protected TestValue createAutogradValue(Supplier<Float> data, Void context,
                boolean tracked, List<? extends ValueNode<?>> inputs) {
            return new TestValue(data, tracked, inputs);
        }
        @Override protected void afterBackward(List<ValueNode<?>> nodes) {
            hooks++;
            callback.accept(nodes);
        }
    }
}
