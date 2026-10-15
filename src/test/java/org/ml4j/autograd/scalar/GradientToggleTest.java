package org.ml4j.autograd.scalar;

import org.ml4j.autograd.scalar.impl.FloatAutogradValueFactoryImpl;


import org.ml4j.autograd.scalar.impl.FloatWrapperAutogradValueFactoryImpl;

import org.junit.jupiter.api.Test;
import org.ml4j.autograd.AutogradValue;
import org.ml4j.autograd.scalar.FloatWrapperAutogradValue;

import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.*;

class GradientToggleTest {
    private final FloatAutogradValueFactory rawFactory =
            new FloatAutogradValueFactoryImpl();

    private final FloatWrapperAutogradValueFactory factory =
            new FloatWrapperAutogradValueFactoryImpl();

    @Test
    void togglingPreservesDataNodesAndStoredGradientWithoutEvaluation() {
        var reads = new AtomicInteger();
        var leaf = rawFactory.create(() -> { reads.incrementAndGet(); return 2f; }, null, true);
        var node = leaf.valueNode();
        var handle = leaf.gradNode();
        var stored = rawFactory.create(7f, false);
        handle.accumulate(stored);
        AutogradValue<FloatAutogradValue, Float, Void> value = leaf;
        value.setRequiresGrad(false);
        value.setRequiresGrad(false);
        assertFalse(value.requiresGrad());
        value.setRequiresGrad(true);
        assertTrue(value.requiresGrad());
        assertEquals(0, reads.get());
        assertSame(node, value.valueNode());
        assertSame(handle, value.gradNode());
        assertSame(stored, handle.gradient().orElseThrow());
        assertNull(node.backwardFunction());
        assertNull(value.context());
        assertEquals(2f, value.data().get());
    }

    @Test
    void futureGraphsUseTheNewFlagAndExistingResultsKeepTheirFlags() {
        var leaf = rawFactory.create(3f, false);
        var inference = leaf.mul(leaf);
        leaf.setRequiresGrad(true);
        assertFalse(inference.requiresGrad());
        var training = leaf.mul(leaf);
        training.backward();
        assertEquals(6f, leaf.gradNode().gradient().orElseThrow().data().get());
        // Complete differentiation before switching to the next graph.
        leaf.gradNode().reset();
        leaf.setRequiresGrad(false);
        assertTrue(training.requiresGrad());
        var nextInference = leaf.mul(leaf);
        assertFalse(nextInference.requiresGrad());
        assertThrows(IllegalStateException.class, nextInference::backward);
        assertTrue(leaf.gradNode().gradient().isEmpty());
        leaf.setRequiresGrad(true);
        leaf.neg().backward();
        assertEquals(-1f, leaf.gradNode().gradient().orElseThrow().data().get());
    }

    @Test
    void binaryAndUnaryResultsRejectChangesIncludingNoOpRequests() {
        var leaf = factory.create(2f, true);
        leaf.setRequiresGrad(false);
        var binary = leaf.add(leaf);
        var unary = leaf.neg();
        for (var result : new FloatWrapperAutogradValue[]{binary, unary}) {
            var node = result.valueNode();
            var callback = node.backwardFunction();
            assertThrows(IllegalStateException.class, () -> result.setRequiresGrad(false));
            assertThrows(IllegalStateException.class, () -> result.setRequiresGrad(true));
            assertFalse(result.requiresGrad());
            assertSame(node, result.valueNode());
            assertSame(callback, node.backwardFunction());
            assertTrue(result.gradNode().gradient().isEmpty());
        }
    }
}
