package org.ml4j.autograd.node.impl;

import org.junit.jupiter.api.Test;
import org.ml4j.autograd.Accumulatable;
import org.ml4j.autograd.Value;

import java.util.function.Supplier;

import static org.junit.jupiter.api.Assertions.*;

class GradNodeImplTest {
    /** Immutable scalar fixture: lazy, functional addition without autograd machinery. */
    private record TestValue(Supplier<Float> data, boolean failAddition, boolean nullSum)
            implements Value<Float, Void>, Accumulatable<TestValue> {
        TestValue(float value) { this(() -> value, false, false); }
        @Override public Void context() { return null; }
        @Override public TestValue add(TestValue other) {
            if (failAddition) throw new IllegalArgumentException("incompatible contribution");
            if (nullSum) return null;
            return new TestValue(() -> data.get() + other.data.get(), false, false);
        }
    }

    @Test
    void distinguishesAbsentGradientFromStoredZero() {
        var node = new GradNodeImpl<TestValue>();
        assertTrue(node.gradient().isEmpty());
        var zero = new TestValue(0f);
        node.accumulate(zero);
        assertSame(zero, node.gradient().orElseThrow());
        assertEquals(0f, node.gradient().orElseThrow().data().get());
    }

    @Test
    void combinesContributionsWithoutMutatingEarlierValues() {
        var node = new GradNodeImpl<TestValue>();
        var first = new TestValue(2f);
        var second = new TestValue(3f);
        node.accumulate(first);
        var retained = node.gradient().orElseThrow();
        node.accumulate(second);
        assertEquals(5f, node.gradient().orElseThrow().data().get());
        assertEquals(2f, retained.data().get());
        assertEquals(3f, second.data().get());
        node.accumulate(new TestValue(4f));
        assertEquals(9f, node.gradient().orElseThrow().data().get());
        assertEquals(2f, retained.data().get());
    }

    @Test
    void storingReadingCombiningAndResetDoNotDirectlyEvaluateData() {
        var first = new TestValue(() -> { fail("Unexpected data evaluation"); return 2f; }, false, false);
        var second = new TestValue(() -> { fail("Unexpected data evaluation"); return 3f; }, false, false);
        var node = new GradNodeImpl<TestValue>();
        node.accumulate(first);
        assertSame(first, node.gradient().orElseThrow());
        node.accumulate(second);
        assertTrue(node.gradient().isPresent());
        node.reset();
        assertTrue(node.gradient().isEmpty());
    }

    @Test
    void resetIsIdempotentAndNextContributionStartsFresh() {
        var node = new GradNodeImpl<TestValue>();
        var first = new TestValue(2f);
        node.accumulate(first);
        node.reset(); node.reset();
        assertTrue(node.gradient().isEmpty());
        assertEquals(2f, first.data().get());
        var fresh = new TestValue(7f);
        node.accumulate(fresh);
        assertSame(fresh, node.gradient().orElseThrow());
        assertEquals(7f, node.gradient().orElseThrow().data().get());
    }

    @Test
    void nullContributionDoesNotChangeEmptyOrPopulatedState() {
        var node = new GradNodeImpl<TestValue>();
        assertThrows(NullPointerException.class, () -> node.accumulate(null));
        assertTrue(node.gradient().isEmpty());
        var first = new TestValue(2f);
        node.accumulate(first);
        assertThrows(NullPointerException.class, () -> node.accumulate(null));
        assertSame(first, node.gradient().orElseThrow());
    }

    @Test
    void failedAdditionPreservesHeldReference() {
        var node = new GradNodeImpl<TestValue>();
        var first = new TestValue(() -> 2f, true, false);
        node.accumulate(first);
        assertThrows(IllegalArgumentException.class, () -> node.accumulate(new TestValue(3f)));
        assertSame(first, node.gradient().orElseThrow());
        assertEquals(2f, first.data().get());
    }

    @Test
    void nullSumPreservesHeldReference() {
        var node = new GradNodeImpl<TestValue>();
        var first = new TestValue(() -> 2f, false, true);
        node.accumulate(first);
        assertThrows(NullPointerException.class, () -> node.accumulate(new TestValue(3f)));
        assertSame(first, node.gradient().orElseThrow());
    }
}
