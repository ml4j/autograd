package org.ml4j.autograd;

import org.junit.jupiter.api.Test;
import org.ml4j.autograd.scalar.FloatAutogradValue;
import org.ml4j.autograd.scalar.impl.FloatAutogradValueFactoryImpl;

import static org.junit.jupiter.api.Assertions.*;

class AutogradOperationsTest {
    @Test
    void gradientReadsDistinguishAbsenceAndZeroWithoutEvaluatingData() {
        var factory = new FloatAutogradValueFactoryImpl();
        var reads = new java.util.concurrent.atomic.AtomicInteger();
        var input = factory.create(() -> { reads.incrementAndGet(); return 3f; }, null, true);
        var zero = factory.create(() -> { reads.incrementAndGet(); return 0f; }, null, false);
        AutogradOperations<FloatAutogradValue> operations = input;
        assertTrue(operations.gradient().isEmpty());
        input.gradNode().accumulate(zero);
        var retained = operations.gradient().orElseThrow();
        assertSame(input.gradNode().gradient().orElseThrow(), retained);
        assertEquals(0, reads.get());
        assertEquals(0f, retained.data().get().floatValue());
        operations.resetGradient();
        assertTrue(operations.gradient().isEmpty());
        assertEquals(1, reads.get());
        assertEquals(0f, retained.data().get().floatValue());
    }

    @Test
    void gradientReadsExposeAccumulationWhileRetainingHigherDerivatives() {
        var factory = new FloatAutogradValueFactoryImpl();
        var input = factory.create(3f, true);
        var result = input.mul(input);
        AutogradOperations<FloatAutogradValue> operations = input;
        result.backward();
        var retained = operations.gradient().orElseThrow();
        result.backward();
        assertEquals(12f, operations.gradient().orElseThrow().data().get().floatValue());
        assertEquals(6f, retained.data().get().floatValue());
        operations.resetGradient();
        retained.backward();
        assertEquals(2f, operations.gradient().orElseThrow().data().get().floatValue());
    }

    @Test
    void capabilitySupportsTrackingAndBothBackwardEntryPoints() {
        var factory = new FloatAutogradValueFactoryImpl();
        var input = factory.create(3f, false);
        AutogradOperations<FloatAutogradValue> tracking = input;
        assertFalse(tracking.requiresGrad());
        tracking.setRequiresGrad(true);
        assertTrue(tracking.requiresGrad());

        var result = input.mul(input);
        AutogradOperations<FloatAutogradValue> operations = result;
        assertThrows(IllegalStateException.class, () -> operations.setRequiresGrad(false));
        assertThrows(NullPointerException.class, () -> operations.backward(null));
        operations.backward();
        assertEquals(6f, input.gradNode().gradient().orElseThrow().data().get().floatValue());
        operations.backward(factory.create(2f, false));
        assertEquals(18f, input.gradNode().gradient().orElseThrow().data().get().floatValue());
    }
    @Test
    void resetIsLocalAndPreservesRetainedDerivativesAndGraphReuse() {
        var factory = new FloatAutogradValueFactoryImpl();
        var input = factory.create(3f, true);
        var result = input.mul(input);
        result.backward();
        var node = input.gradNode();
        var valueNode = input.valueNode();
        var retained = node.gradient().orElseThrow();
        var resultGradient = result.gradNode().gradient().orElseThrow();
        AutogradOperations<FloatAutogradValue> operations = input;
        operations.resetGradient();
        operations.resetGradient();
        assertTrue(node.gradient().isEmpty());
        assertSame(node, input.gradNode());
        assertSame(valueNode, input.valueNode());
        assertTrue(input.requiresGrad());
        assertEquals(3f, input.data().get().floatValue());
        assertSame(resultGradient, result.gradNode().gradient().orElseThrow());
        assertEquals(6f, retained.data().get().floatValue());

        result.backward();
        assertEquals(6f, node.gradient().orElseThrow().data().get().floatValue());
        operations.resetGradient();
        retained.backward();
        assertEquals(2f, node.gradient().orElseThrow().data().get().floatValue());

        var inputGradient = node.gradient().orElseThrow();
        result.resetGradient();
        assertTrue(result.gradNode().gradient().isEmpty());
        assertSame(inputGradient, node.gradient().orElseThrow());
    }

    @Test
    void resetDoesNotEvaluateForwardOrGradientDataOrChangeTracking() {
        var factory = new FloatAutogradValueFactoryImpl();
        var reads = new java.util.concurrent.atomic.AtomicInteger();
        var input = factory.create(() -> { reads.incrementAndGet(); return 3f; }, null, false);
        var contribution = factory.create(() -> { reads.incrementAndGet(); return 7f; }, null, false);
        input.gradNode().accumulate(contribution);
        AutogradOperations<FloatAutogradValue> operations = input;
        operations.resetGradient();
        operations.resetGradient();
        assertTrue(input.gradNode().gradient().isEmpty());
        assertFalse(operations.requiresGrad());
        assertEquals(0, reads.get());
        assertEquals(7f, contribution.data().get().floatValue());
    }
}
