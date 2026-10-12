package org.ml4j.autograd.scalar;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.*;

class MultiplicationTest {
    private FloatAutogradValue value(float data, boolean requiresGrad) {
        return new FloatAutogradValue(data, requiresGrad);
    }
    private float gradient(FloatAutogradValue value) {
        return value.gradNode().gradient().orElseThrow().data().get();
    }

    @Test void productIsLazyAndPreservesOperandStructure() {
        var reads = new AtomicInteger();
        var left = new FloatAutogradValue(() -> { reads.incrementAndGet(); return 2f; }, true);
        var right = value(3f, true);
        var product = left.mul(right);
        assertEquals(0, reads.get());
        assertEquals(List.of(left.valueNode(), right.valueNode()), product.valueNode().inputs());
        assertEquals(6f, product.data().get());
        assertEquals(2f, left.data().get());
        assertEquals(3f, right.data().get());
        assertNull(left.valueNode().backwardFunction());
        assertTrue(left.gradNode().gradient().isEmpty());
        assertThrows(NullPointerException.class, () -> left.mul(null));
    }

    @Test void explicitSeedScalesBothPartialDerivatives() {
        var left = value(2f, true); var right = value(3f, true);
        left.mul(right).backward(value(4f, false));
        assertEquals(12f, gradient(left));
        assertEquals(8f, gradient(right));
    }

    @Test void constantsReceiveNoGradientInEitherOperandPosition() {
        for (boolean trackedOnLeft : new boolean[]{false, true}) {
            var tracked = value(2f, true); var constant = value(3f, false);
            var product = trackedOnLeft ? tracked.mul(constant) : constant.mul(tracked);
            assertTrue(product.requiresGrad());
            product.backward();
            assertEquals(3f, gradient(tracked));
            assertTrue(constant.gradNode().gradient().isEmpty());
        }
        assertFalse(value(2f, false).mul(value(3f, false)).requiresGrad());
    }

    @Test void zeroPartialDerivativeRemainsAPresentGradient() {
        var left = value(2f, true); var right = value(0f, true);
        left.mul(right).backward();
        assertTrue(left.gradNode().gradient().isPresent());
        assertEquals(0f, gradient(left));
        assertEquals(2f, gradient(right));
    }

    @Test void repeatedOperandsAndSharedBranchesAccumulateWithoutRepropagatingOldGradients() {
        var x = value(2f, true); var square = x.mul(x); var output = square.add(square);
        assertEquals(List.of(x.valueNode(), x.valueNode()), square.valueNode().inputs());
        output.backward();
        var earlier = x.gradNode().gradient().orElseThrow();
        assertEquals(8f, gradient(x));
        output.backward();
        assertEquals(16f, gradient(x));
        assertEquals(2f, gradient(square));
        assertEquals(8f, earlier.data().get());
        x.gradNode().reset();
        output.backward();
        assertEquals(8f, gradient(x));
    }

    @Test void retainedSquareGradientCanBeDifferentiatedAfterReset() {
        var x = value(3f, true); var handle = x.gradNode();
        x.mul(x).backward();
        var firstDerivative = handle.gradient().orElseThrow();
        assertEquals(6f, firstDerivative.data().get());
        handle.reset();
        firstDerivative.backward();
        assertSame(handle, x.gradNode());
        assertEquals(2f, gradient(x));
        assertEquals(6f, firstDerivative.data().get());
    }

    @Test void hessianVectorMatchesProofOfConceptWithAndWithoutLeafReset() {
        for (boolean reset : new boolean[]{false, true}) {
            var x = value(0.5f, true); var y = value(0.6f, true);
            var z = x.mul(x).add(y.mul(x)).add(y.mul(y));
            z.backward();
            var gx = x.gradNode().gradient().orElseThrow();
            var gy = y.gradNode().gradient().orElseThrow();
            assertEquals(1.6f, gx.data().get(), 0.0001f);
            assertEquals(1.7f, gy.data().get(), 0.0001f);
            if (reset) { x.gradNode().reset(); y.gradNode().reset(); }
            gx.mul(value(2f, false)).add(gy).backward();
            assertEquals(reset ? 5f : 6.6f, gradient(x), 0.0001f);
            assertEquals(reset ? 4f : 5.7f, gradient(y), 0.0001f);
            assertEquals(1.6f, gx.data().get(), 0.0001f);
            assertEquals(1.7f, gy.data().get(), 0.0001f);
        }
    }
}
