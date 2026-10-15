package org.ml4j.autograd.scalar.impl;

import org.junit.jupiter.api.Test;
import org.ml4j.autograd.scalar.FloatAutogradValue;

import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.*;

class FloatAutogradValueTest {
    @Test
    void leafOwnsStableNodesWithEmptyGradientAndNoInputs() {
        var value = new FloatAutogradValueImpl(2f, true);
        assertEquals(2f, value.data().get());
        assertNull(value.context());
        assertSame(value, value.valueNode().value());
        assertSame(value.valueNode(), value.valueNode());
        var handle = value.gradNode();
        handle.reset();
        assertSame(handle, value.gradNode());
        assertTrue(handle.gradient().isEmpty());
        assertTrue(value.valueNode().inputs().isEmpty());
    }

    @Test
    void additionIsLazyAndRecordsOrderedInputsWithoutMutatingOperands() {
        var reads = new AtomicInteger();
        var first = new FloatAutogradValueImpl(() -> { reads.incrementAndGet(); return 2f; }, false);
        var second = new FloatAutogradValueImpl(3f, true);
        var sum = first.add(second);
        assertEquals(0, reads.get());
        assertSame(sum, sum.valueNode().value());
        assertEquals(List.of(first.valueNode(), second.valueNode()), sum.valueNode().inputs());
        assertEquals(0, reads.get());
        assertEquals(5f, sum.data().get());
        assertEquals(5f, sum.data().get());
        assertEquals(2f, first.data().get());
        assertNull(first.valueNode().backwardFunction());
        assertNull(second.valueNode().backwardFunction());
        assertTrue(first.gradNode().gradient().isEmpty());
        assertTrue(second.gradNode().gradient().isEmpty());
        assertNotNull(sum.valueNode().backwardFunction());
    }

    @Test
    void additionCombinesBothGradientRequirements() {
        for (boolean left : new boolean[]{false, true}) {
            for (boolean right : new boolean[]{false, true}) {
                var first = new FloatAutogradValueImpl(2f, left);
                var second = new FloatAutogradValueImpl(3f, right);
                assertEquals(left || right, first.add(second).requiresGrad());
            }
        }
    }

    @Test
    void repeatedOperandsAndChainedResultsRetainTheirGraph() {
        var x = new FloatAutogradValueImpl(2f, true);
        var doubled = x.add(x);
        assertEquals(List.of(x.valueNode(), x.valueNode()), doubled.valueNode().inputs());
        assertThrows(UnsupportedOperationException.class, () -> doubled.valueNode().inputs().clear());
        var result = doubled.add(x);
        assertEquals(6f, result.data().get());
        assertEquals(List.of(doubled.valueNode(), x.valueNode()), result.valueNode().inputs());
    }

    @Test
    void localBackwardWithoutSeedDoesNotChangeGradientState() {
        var value = new FloatAutogradValueImpl(2f, true);
        var sum = value.add(value);
        sum.valueNode().backward(); // No incoming gradient: local propagation does nothing.
        assertTrue(sum.gradNode().gradient().isEmpty());
        assertTrue(value.gradNode().gradient().isEmpty());
    }

    @Test
    void nullInputsAreRejectedWithoutEagerNumericalEvaluation() {
        assertThrows(NullPointerException.class, () -> new FloatAutogradValueImpl(null, false));
        var value = new FloatAutogradValueImpl(2f, false);
        assertThrows(NullPointerException.class, () -> value.add(null));
        var invalid = new FloatAutogradValueImpl(() -> null, false);
        assertThrows(NullPointerException.class, () -> invalid.data().get());
    }
}
