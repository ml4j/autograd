package org.ml4j.autograd.activation;

import org.junit.jupiter.api.Test;
import org.ml4j.autograd.scalar.FloatWrapperAutogradValue;
import org.ml4j.autograd.scalar.FloatWrapperAutogradValueImpl;
import org.ml4j.autograd.scalar.FloatWrapperImpl;

import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.*;

class ReluTest {
    private FloatWrapperAutogradValue value(float number, boolean tracked) {
        return new FloatWrapperAutogradValueImpl(number, tracked);
    }

    @Test
    void numericalReluPreservesInputAndHandlesFloatBoundaries() {
        for (float number : new float[]{-3f, 0f, -0f, 2f, Float.NEGATIVE_INFINITY, Float.POSITIVE_INFINITY, Float.NaN}) {
            var input = new FloatWrapperImpl(number);
            var output = input.relu();
            assertNotSame(input, output);
            assertEquals(Float.floatToIntBits(Math.max(0f, number)), Float.floatToIntBits(output.floatValue()));
            assertEquals(Float.floatToIntBits(number), Float.floatToIntBits(input.floatValue()));
        }
    }

    @Test
    void constructionIsLazyAndPreservesNodesContextAndStoredGradient() {
        var reads = new AtomicInteger();
        var input = new FloatWrapperAutogradValueImpl(() -> {
            reads.incrementAndGet(); return new FloatWrapperImpl(2f);
        }, true);
        var stored = value(7f, false);
        input.gradNode().accumulate(stored);
        var output = input.relu();
        assertEquals(0, reads.get());
        assertEquals(1, output.valueNode().inputs().size());
        assertSame(input.valueNode(), output.valueNode().inputs().getFirst());
        assertNull(output.context());
        assertTrue(output.requiresGrad());
        assertTrue(output.gradNode().gradient().isEmpty());
        output.valueNode().backward();
        assertSame(stored, input.gradNode().gradient().orElseThrow());
        assertEquals(0, reads.get());
        assertEquals(2f, output.data().get().floatValue());
    }

    @Test
    void seededDerivativesAreZeroAtAndBelowZeroAndConstantsStayUntracked() {
        for (float number : new float[]{-2f, -0f, 0f, 3f}) {
            var input = value(number, true);
            input.relu().backward(value(4f, false));
            assertEquals(number > 0 ? 4f : 0f, input.gradNode().gradient().orElseThrow().data().get().floatValue());
            var constant = value(number, false);
            var result = constant.relu();
            assertFalse(result.requiresGrad());
            assertThrows(IllegalStateException.class, result::backward);
            assertTrue(constant.gradNode().gradient().isEmpty());
        }
    }

    @Test
    void sharedBranchesRepeatedPassesAndResetProduceFreshContributions() {
        var input = value(2f, true);
        var branch = input.relu();
        var result = branch.add(branch);
        result.backward(value(1f, false));
        assertEquals(2f, input.gradNode().gradient().orElseThrow().data().get().floatValue());
        result.backward(value(1f, false));
        assertEquals(4f, input.gradNode().gradient().orElseThrow().data().get().floatValue());
        input.gradNode().reset();
        result.backward(value(1f, false));
        assertEquals(2f, input.gradNode().gradient().orElseThrow().data().get().floatValue());
    }

    @Test
    void composedHigherDerivativesRetainIncomingDependenciesAndUseFixedMasks() {
        for (float number : new float[]{-2f, 0f, 3f}) {
            var input = value(number, true);
            var activated = input.relu();
            activated.mul(activated).backward(value(1f, false));
            var first = input.gradNode().gradient().orElseThrow();
            assertEquals(number > 0 ? 2 * number : 0f, first.data().get().floatValue());
            input.gradNode().reset();
            first.backward(value(1f, false));
            assertEquals(number > 0 ? 2f : 0f, input.gradNode().gradient().orElseThrow().data().get().floatValue());
        }
    }

    @Test
    void microgradDemoMatchesExpectedForwardAndInputGradients() {
        var a = value(-4f, true);
        var b = value(2f, true);
        var c = a.add(b);
        var d = a.mul(b).add(b.mul(b).mul(b));
        c = c.add(c.add(1f));
        c = c.add(value(1f, false).add(c).add(a.neg()));
        d = d.add(d.mul(2f).add(b.add(a).relu()));
        d = d.add(value(3f, false).mul(d).add(b.sub(a).relu()));
        var e = c.sub(d);
        var f = e.mul(e);
        var g = f.div(2f).add(value(10f, false).div(f));
        assertEquals(24.704082f, g.data().get().floatValue(), 1e-5f);
        g.backward();
        assertEquals(138.83382f, a.gradNode().gradient().orElseThrow().data().get().floatValue(), 1e-4f);
        assertEquals(645.5772f, b.gradNode().gradient().orElseThrow().data().get().floatValue(), 1e-3f);
    }
}
