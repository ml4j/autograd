package org.ml4j.autograd.scalar;

import org.junit.jupiter.api.Test;
import org.ml4j.autograd.arithmetic.scalar.FloatWrapperAutogradValueImpl;
import org.ml4j.autograd.arithmetic.scalar.FloatWrapperImpl;

import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.*;

class SubtractionTest {
    @Test void rawSubtractionIsLazyAndDoesNotMutateOperands() {
        var reads = new AtomicInteger();
        var left = new FloatAutogradValue(() -> { reads.incrementAndGet(); return 5f; }, true);
        var right = new FloatAutogradValue(() -> { reads.incrementAndGet(); return 3f; }, true);
        var result = left.sub(right);
        assertEquals(0, reads.get());
        assertEquals(2f, result.data().get());
        assertEquals(2, reads.get());
        result.backward(new FloatAutogradValue(4f, false));
        assertEquals(4f, left.gradNode().gradient().orElseThrow().data().get());
        assertEquals(-4f, right.gradNode().gradient().orElseThrow().data().get());
        assertEquals(5f, left.data().get());
        assertEquals(3f, right.data().get());
    }

    @Test void wrappedDefaultRetainsCompositionAndSignedGradients() {
        var left = new FloatWrapperAutogradValueImpl(5f, true);
        var right = new FloatWrapperAutogradValueImpl(3f, true);
        var result = left.sub(right);
        assertSame(left.valueNode(), result.valueNode().inputs().getFirst());
        var negation = result.valueNode().inputs().get(1);
        assertEquals(1, negation.inputs().size());
        assertSame(right.valueNode(), negation.inputs().getFirst());
        assertEquals(2f, result.data().get().getFloat());
        result.backward(new FloatWrapperAutogradValueImpl(4f, false));
        assertEquals(4f, left.gradNode().gradient().orElseThrow().data().get().getFloat());
        assertEquals(-4f, right.gradNode().gradient().orElseThrow().data().get().getFloat());
    }

    @Test void selfSubtractionCancelsAndRepeatedPassesRefreshIntermediates() {
        var x = new FloatAutogradValue(3f, true);
        var cancellation = x.sub(x);
        assertEquals(0f, cancellation.data().get());
        cancellation.backward();
        assertEquals(0f, x.gradNode().gradient().orElseThrow().data().get());
        cancellation.backward();
        assertEquals(0f, x.gradNode().gradient().orElseThrow().data().get());
        var right = new FloatAutogradValue(2f, true);
        var difference = x.sub(right);
        difference.backward();
        var firstRightGradient = right.gradNode().gradient().orElseThrow();
        difference.backward();
        assertEquals(2f, x.gradNode().gradient().orElseThrow().data().get());
        assertEquals(-2f, right.gradNode().gradient().orElseThrow().data().get());
        assertEquals(-1f, firstRightGradient.data().get());
    }

    @Test void constantPositionsAreSkippedAndRequirementsAreCombined() {
        var constant = new FloatWrapperAutogradValueImpl(5f, false);
        var tracked = new FloatWrapperAutogradValueImpl(3f, true);
        var result = constant.sub(tracked);
        assertTrue(result.requiresGrad());
        result.backward();
        assertTrue(constant.gradNode().gradient().isEmpty());
        assertEquals(-1f, tracked.gradNode().gradient().orElseThrow().data().get().getFloat());
        tracked.gradNode().reset();
        tracked.sub(constant).backward();
        assertTrue(constant.gradNode().gradient().isEmpty());
        assertEquals(1f, tracked.gradNode().gradient().orElseThrow().data().get().getFloat());
        var constantResult = constant.sub(new FloatWrapperAutogradValueImpl(2f, false));
        assertFalse(constantResult.requiresGrad());
        assertEquals(3f, constantResult.data().get().getFloat());
    }

    @Test void compositionWithMultiplicationPreservesSecondDerivatives() {
        var x = new FloatAutogradValue(3f, true);
        x.mul(x).sub(x).backward(); // x²-x: first derivative 5, second derivative 2.
        var first = x.gradNode().gradient().orElseThrow();
        assertEquals(5f, first.data().get());
        x.gradNode().reset();
        first.backward();
        assertEquals(2f, x.gradNode().gradient().orElseThrow().data().get());

        var wrapped = new FloatWrapperAutogradValueImpl(3f, true);
        wrapped.sub(wrapped.mul(wrapped)).backward(); // x-x²: first -5, second -2.
        var wrappedFirst = wrapped.gradNode().gradient().orElseThrow();
        assertEquals(-5f, wrappedFirst.data().get().getFloat());
        wrapped.gradNode().reset();
        wrappedFirst.backward();
        assertEquals(-2f, wrapped.gradNode().gradient().orElseThrow().data().get().getFloat());
    }

    @Test void numericalSubtractionAndNullOperandsFollowTheContract() {
        var left = new FloatWrapperImpl(5f);
        var right = new FloatWrapperImpl(3f);
        assertEquals(2f, left.sub(right).getFloat());
        assertEquals(5f, left.getFloat());
        assertEquals(3f, right.getFloat());
        assertThrows(NullPointerException.class, () -> left.sub(null));
        var raw = new FloatAutogradValue(5f, true);
        var wrapped = new FloatWrapperAutogradValueImpl(5f, true);
        assertThrows(NullPointerException.class, () -> raw.sub(null));
        assertThrows(NullPointerException.class, () -> wrapped.sub(null));
        assertTrue(raw.gradNode().gradient().isEmpty());
        assertTrue(wrapped.gradNode().gradient().isEmpty());
    }
}
