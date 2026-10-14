package org.ml4j.autograd.math;

import org.junit.jupiter.api.Test;
import org.ml4j.autograd.arithmetic.scalar.FloatWrapperAutogradValue;
import org.ml4j.autograd.arithmetic.scalar.FloatWrapperAutogradValueImpl;
import org.ml4j.autograd.arithmetic.scalar.FloatWrapperImpl;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.UnaryOperator;
import static org.junit.jupiter.api.Assertions.*;

class MathOperationsTest {
    private FloatWrapperAutogradValue value(float x, boolean tracked) {
        return new FloatWrapperAutogradValueImpl(x, tracked);
    }

    private final java.util.List<UnaryOperator<FloatWrapperAutogradValue>> functions = java.util.List.of(
            FloatWrapperAutogradValue::exp, FloatWrapperAutogradValue::log, x -> x.pow(3f));

    @Test
    void numericalFunctionsPreserveInputsAndFollowJavaBoundaries() {
        for (float x : new float[]{-2f, -0f, 0f, 2f, 100f, Float.NaN,
                Float.NEGATIVE_INFINITY, Float.POSITIVE_INFINITY}) {
            var input = new FloatWrapperImpl(x);
            assertEquals((float) Math.exp(x), input.exp().floatValue());
            assertEquals((float) Math.log(x), input.log().floatValue());
            for (float p : new float[]{-1f, 0f, 1f, 0.5f, 2f, Float.NaN, Float.POSITIVE_INFINITY}) {
                assertEquals((float) Math.pow(x, p), input.pow(p).floatValue());
            }
            assertEquals(Float.floatToIntBits(x), Float.floatToIntBits(input.floatValue()));
        }
    }

    @Test
    void allFunctionsAreLazyAndPreserveOperandStateAndConstants() {
        for (var function : functions) {
            var reads = new AtomicInteger();
            var input = new FloatWrapperAutogradValueImpl(() -> {
                reads.incrementAndGet();
                return new FloatWrapperImpl(2f);
            }, true);
            var stored = value(7f, false);
            input.gradNode().accumulate(stored);
            var output = function.apply(input);
            assertEquals(0, reads.get());
            assertEquals(1, output.valueNode().inputs().size());
            assertSame(input.valueNode(), output.valueNode().inputs().getFirst());
            assertNull(output.context());
            assertTrue(output.requiresGrad());
            assertTrue(output.gradNode().gradient().isEmpty());
            assertSame(stored, input.gradNode().gradient().orElseThrow());
            output.data().get();
            assertTrue(reads.get() > 0);
            var constant = value(2f, false);
            var result = function.apply(constant);
            assertFalse(result.requiresGrad());
            assertThrows(IllegalStateException.class, result::backward);
            assertTrue(constant.gradNode().gradient().isEmpty());
        }
    }

    @Test
    void seededSharedRepeatedAndSecondDerivatives() {
        double[] firstExpected = {Math.exp(2), 0.5, 12};
        double[] secondExpected = {Math.exp(2), -0.25, 12};
        for (int i = 0; i < functions.size(); i++) {
            var input = value(2f, true);
            var branch = functions.get(i).apply(input);
            var output = branch.add(branch);
            output.backward(value(3f, false));
            var first = input.gradNode().gradient().orElseThrow();
            assertEquals(6 * firstExpected[i], first.data().get().floatValue(), 1e-4);
            output.backward(value(3f, false));
            assertEquals(12 * firstExpected[i], input.gradNode().gradient().orElseThrow().data().get().floatValue(), 1e-4);
            input.gradNode().reset();
            first.backward();
            assertEquals(6 * secondExpected[i], input.gradNode().gradient().orElseThrow().data().get().floatValue(), 1e-4);
        }
    }

    @Test
    void powerKeepsGeneralRuleIncludingZeroAndOneBoundaryEffects() {
        for (float p : new float[]{0f, 1f, -0.5f, 0.5f}) {
            var input = value(2f, true);
            input.pow(p).backward();
            assertEquals(p * Math.pow(2, p - 1), input.gradNode().gradient().orElseThrow().data().get().floatValue(), 1e-6);
        }
        var zero = value(0f, true);
        assertEquals(1f, zero.pow(0f).data().get().floatValue());
        zero.pow(0f).backward();
        assertTrue(Float.isNaN(zero.gradNode().gradient().orElseThrow().data().get().floatValue()));
        var negative = value(-2f, true);
        negative.pow(3f).backward();
        assertEquals(12f, negative.gradNode().gradient().orElseThrow().data().get().floatValue());
    }
}
