package org.ml4j.autograd.activation;

import org.junit.jupiter.api.Test;
import org.ml4j.autograd.scalar.FloatWrapperAutogradValue;
import org.ml4j.autograd.scalar.FloatWrapperAutogradValueImpl;
import org.ml4j.autograd.scalar.FloatWrapperImpl;
import java.util.concurrent.atomic.AtomicInteger;
import static org.junit.jupiter.api.Assertions.*;

class SigmoidTest {
    private FloatWrapperAutogradValue value(float x, boolean tracked) {
        return new FloatWrapperAutogradValueImpl(x, tracked);
    }

    @Test
    void numericalValuesPreserveInputAndHandleBoundaries() {
        for (float x : new float[]{-100f, -2f, -0f, 0f, 2f, 100f,
                Float.NEGATIVE_INFINITY, Float.POSITIVE_INFINITY, Float.NaN}) {
            var input = new FloatWrapperImpl(x);
            var output = input.sigmoid();
            assertNotSame(input, output);
            assertEquals((float) (1d / (1d + Math.exp(-(double) x))), output.floatValue(), 1e-7f);
            assertEquals(Float.floatToIntBits(x), Float.floatToIntBits(input.floatValue()));
        }
        assertTrue(new FloatWrapperImpl(-100f).sigmoid().floatValue() > 0f);
    }

    @Test
    void constructionIsLazyAndPreservesGraphAndGradientState() {
        var reads = new AtomicInteger();
        var input = new FloatWrapperAutogradValueImpl(() -> {
            reads.incrementAndGet();
            return new FloatWrapperImpl(0f);
        }, true);
        var stored = value(7f, false);
        input.gradNode().accumulate(stored);
        var output = input.sigmoid();
        assertEquals(0, reads.get());
        assertEquals(1, output.valueNode().inputs().size());
        assertSame(input.valueNode(), output.valueNode().inputs().getFirst());
        assertNull(output.context());
        assertTrue(output.requiresGrad());
        assertTrue(output.gradNode().gradient().isEmpty());
        assertSame(stored, input.gradNode().gradient().orElseThrow());
        assertEquals(0.5f, output.data().get().floatValue());
    }

    @Test
    void seededDerivativesAndConstants() {
        for (float x : new float[]{-2f, 0f, 2f, Float.NEGATIVE_INFINITY, Float.POSITIVE_INFINITY}) {
            var input = value(x, true);
            input.sigmoid().backward(value(4f, false));
            double s = 1d / (1d + Math.exp(-x));
            assertEquals(4 * s * (1 - s), input.gradNode().gradient().orElseThrow().data().get().floatValue(), 1e-6);
            var constant = value(x, false);
            var result = constant.sigmoid();
            assertFalse(result.requiresGrad());
            assertThrows(IllegalStateException.class, result::backward);
            assertTrue(constant.gradNode().gradient().isEmpty());
        }
    }

    @Test
    void sharedBranchesRepeatedPassesAndReset() {
        var input = value(0f, true);
        var branch = input.sigmoid();
        var output = branch.add(branch);
        output.backward();
        assertEquals(0.5f, input.gradNode().gradient().orElseThrow().data().get().floatValue());
        output.backward();
        assertEquals(1f, input.gradNode().gradient().orElseThrow().data().get().floatValue());
        input.gradNode().reset();
        output.backward();
        assertEquals(0.5f, input.gradNode().gradient().orElseThrow().data().get().floatValue());
    }

    @Test
    void secondDerivativesRetainOperandDependencies() {
        for (float x : new float[]{-2f, 0f, 2f}) {
            var input = value(x, true);
            input.sigmoid().backward();
            var first = input.gradNode().gradient().orElseThrow();
            input.gradNode().reset();
            first.backward();
            double s = 1d / (1d + Math.exp(-x));
            assertEquals(s * (1 - s) * (1 - 2 * s),
                    input.gradNode().gradient().orElseThrow().data().get().floatValue(), 1e-6);
        }
    }
}
