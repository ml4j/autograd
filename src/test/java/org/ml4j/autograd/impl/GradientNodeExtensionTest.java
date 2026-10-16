package org.ml4j.autograd.impl;

import org.junit.jupiter.api.Test;
import org.ml4j.autograd.node.GradNode;
import org.ml4j.autograd.node.ValueNode;
import org.ml4j.autograd.node.impl.GradNodeImpl;

import java.util.List;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Supplier;

import static org.junit.jupiter.api.Assertions.*;

/** Backend-independent evidence for supplying an extensible gradient node. */
class GradientNodeExtensionTest {
    @Test
    void suppliedNodeIsRetainedWithoutConstructorCallsOrDataEvaluation() {
        AtomicInteger reads = new AtomicInteger();
        ViewGradNode node = new ViewGradNode();
        TestValue value = new TestValue(() -> { reads.incrementAndGet(); return 3f; }, true, node);
        assertSame(node, value.gradNode());
        assertSame(value.gradNode(), value.gradNode());
        assertEquals(0, node.reads);
        assertEquals(0, reads.get());
        assertEquals(3f, value.data().get());
        assertEquals(1, reads.get());
    }

    @Test
    void existingConstructorStillCreatesIndependentDefaultNodes() {
        TestValue first = new TestValue(2, true), second = new TestValue(3, true);
        assertInstanceOf(GradNodeImpl.class, first.gradNode());
        assertNotSame(first.gradNode(), second.gradNode());
        assertTrue(first.gradNode().gradient().isEmpty());
        first.gradNode().accumulate(new TestValue(4, false));
        assertEquals(4f, first.gradNode().gradient().orElseThrow().data().get());
        assertTrue(second.gradNode().gradient().isEmpty());
    }

    @Test
    void nullSuppliedNodeFailsWithoutEvaluatingData() {
        assertThrows(NullPointerException.class, () -> new TestValue(
            () -> { throw new AssertionError("Constructor must not evaluate data"); }, true, (GradNode<TestValue>) null));
    }

    @Test
    void overriddenReadViewsDoNotReplaceTheUnderlyingAccumulationState() {
        ViewGradNode node = new ViewGradNode();
        TestValue first = new TestValue(2, false), second = new TestValue(3, false);
        node.accumulate(first);
        assertEquals(0, node.reads, "inherited accumulation must not call overridden gradient()");
        TestValue oldView = node.gradient().orElseThrow();
        assertSame(first, node.raw().orElseThrow());
        assertSame(oldView, node.gradient().orElseThrow());
        node.accumulate(second);
        assertEquals(5f, node.raw().orElseThrow().data().get());
        assertEquals(5f, node.gradient().orElseThrow().data().get());
        assertEquals(2f, oldView.data().get());
        TestValue held = node.raw().orElseThrow();
        assertThrows(NullPointerException.class, () -> node.accumulate(null));
        assertSame(held, node.raw().orElseThrow());
        int previousReads = node.reads;
        node.reset();
        assertEquals(previousReads, node.reads, "inherited reset must not call overridden gradient()");
        assertTrue(node.gradient().isEmpty());
        node.accumulate(new TestValue(0, false));
        assertEquals(0f, node.gradient().orElseThrow().data().get());
        assertEquals(2f, oldView.data().get());
    }

    @Test
    void suppliedReadViewsPreserveBackwardAccumulationResetAndDuplicateOperands() {
        ViewGradNode node = new ViewGradNode();
        TestValue x = new TestValue(() -> 2f, true, node);
        TestValue frozen = new TestValue(3, false);
        TestValue result = x.mul(x).add(x).add(frozen);
        TestValue seed = new TestValue(2, false);
        result.backward(seed);
        TestValue previous = x.gradNode().gradient().orElseThrow();
        assertEquals(10f, previous.data().get());
        assertTrue(node.reads > 0);
        assertTrue(frozen.gradNode().gradient().isEmpty());
        result.backward(seed);
        assertSame(node, x.gradNode());
        assertEquals(20f, node.gradient().orElseThrow().data().get());
        assertEquals(10f, previous.data().get());
        node.reset();
        result.backward(seed);
        assertEquals(10f, node.gradient().orElseThrow().data().get());
    }

    @Test
    void suppliedReadViewsStayLazyAndRetainHigherDerivativeGraphs() {
        AtomicInteger dataReads = new AtomicInteger();
        ViewGradNode node = new ViewGradNode();
        TestValue x = new TestValue(() -> { dataReads.incrementAndGet(); return 3f; }, true, node);
        x.mul(x).mul(x).backward(new TestValue(1, false));
        TestValue first = node.gradient().orElseThrow();
        assertEquals(0, dataReads.get());
        assertTrue(first.requiresGrad());
        assertEquals(27f, first.data().get());
        node.reset();
        first.backward(new TestValue(1, false));
        assertEquals(18f, node.gradient().orElseThrow().data().get());
        assertEquals(27f, first.data().get());
    }

    /** Equivalent graph-bearing read view; this library has no native/backend dependency. */
    private static final class ViewGradNode extends GradNodeImpl<TestValue> {
        private TestValue original;
        private TestValue view;
        private int reads;

        @Override
        public Optional<TestValue> gradient() {
            reads++;
            Optional<TestValue> stored = super.gradient();
            if (stored.isEmpty()) {
                original = null;
                view = null;
                return Optional.empty();
            }
            if (original != stored.get()) {
                original = stored.get();
                view = original.add(new TestValue(0, false));
            }
            return Optional.of(view);
        }

        Optional<TestValue> raw() { return super.gradient(); }
    }

    private static final class TestValue extends AutogradValueImpl<TestValue, Float, Void> {
        TestValue(float value, boolean tracked) { this(() -> value, tracked, List.of()); }
        TestValue(Supplier<Float> data, boolean tracked, List<? extends ValueNode<?>> inputs) {
            super(data, null, tracked, inputs);
        }
        TestValue(Supplier<Float> data, boolean tracked, GradNode<TestValue> gradNode) {
            super(data, null, tracked, List.of(), gradNode);
        }
        @Override protected TestValue self() { return this; }
        @Override public Void mapContext(Void left, Void right) { return null; }
        @Override protected TestValue createAutogradValue(Supplier<Float> data, Void context,
                boolean tracked, List<? extends ValueNode<?>> inputs) {
            return new TestValue(data, tracked, inputs);
        }
        @Override public TestValue add(TestValue other) {
            return applyBinaryOperator(other, (a, b) -> a + b,
                (g, a, b) -> g, (g, a, b) -> g, this::mapContext);
        }
        TestValue mul(TestValue other) {
            return applyBinaryOperator(other, (a, b) -> a * b,
                (g, a, b) -> g.mul(b), (g, a, b) -> g.mul(a), this::mapContext);
        }
    }
}
