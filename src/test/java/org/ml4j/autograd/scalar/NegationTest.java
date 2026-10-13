package org.ml4j.autograd.scalar;

import org.junit.jupiter.api.Test;
import org.ml4j.autograd.arithmetic.scalar.FloatWrapperAutogradValueImpl;
import org.ml4j.autograd.arithmetic.scalar.FloatWrapperImpl;

import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.*;

class NegationTest {
    @Test void rawNegationIsLazyAndPreservesOperandState() {
        var reads = new AtomicInteger();
        var x = new FloatAutogradValue(() -> { reads.incrementAndGet(); return 2f; }, true);
        var result = x.neg();
        assertEquals(0, reads.get());
        assertTrue(result.requiresGrad());
        assertNull(result.context());
        assertEquals(1, result.valueNode().inputs().size());
        assertSame(x.valueNode(), result.valueNode().inputs().getFirst());
        assertEquals(-2f, result.data().get());
        assertEquals(1, reads.get());
        assertTrue(x.gradNode().gradient().isEmpty());
        assertEquals(2f, x.data().get());
    }

    @Test void explicitSeedsAndRepeatedPassesAccumulateOnlyNewContributions() {
        var x = new FloatAutogradValue(2f, true);
        var result = x.neg();
        var seed = new FloatAutogradValue(3f, false);
        result.backward(seed);
        var first = x.gradNode().gradient().orElseThrow();
        assertEquals(-3f, first.data().get());
        result.backward(seed);
        assertEquals(-6f, x.gradNode().gradient().orElseThrow().data().get());
        assertEquals(-3f, first.data().get());
        assertEquals(3f, seed.data().get());
        x.gradNode().reset();
        result.backward();
        assertEquals(-1f, x.gradNode().gradient().orElseThrow().data().get());
    }

    @Test void rawNegatedSquareSupportsSecondDerivatives() {
        var x = new FloatAutogradValue(3f, true);
        var result = x.mul(x).neg(); // -x²: value -9, first derivative -6, second -2.
        assertEquals(-9f, result.data().get());
        result.backward();
        var first = x.gradNode().gradient().orElseThrow();
        assertEquals(-6f, first.data().get());
        assertTrue(first.requiresGrad());
        x.gradNode().reset();
        first.backward();
        assertEquals(-2f, x.gradNode().gradient().orElseThrow().data().get());
    }

    @Test void wrappedDefaultSupportsNumericalNegationAndSecondDerivatives() {
        var data = new FloatWrapperImpl(3f);
        assertEquals(-3f, data.neg().floatValue());
        assertEquals(3f, data.floatValue());
        var x = new FloatWrapperAutogradValueImpl(data, true);
        var result = x.neg().mul(x); // (-x)*x exercises unary propagation inside a product.
        assertEquals(-9f, result.data().get().floatValue());
        result.backward();
        var first = x.gradNode().gradient().orElseThrow();
        assertEquals(-6f, first.data().get().floatValue());
        x.gradNode().reset();
        first.backward();
        assertEquals(-2f, x.gradNode().gradient().orElseThrow().data().get().floatValue());
    }

    @Test void constantNegationReceivesNoLocalContribution() {
        var raw = new FloatAutogradValue(2f, false);
        var result = raw.neg();
        assertFalse(result.requiresGrad());
        result.gradNode().accumulate(new FloatAutogradValue(1f, false));
        result.valueNode().backward();
        assertTrue(raw.gradNode().gradient().isEmpty());
        assertThrows(IllegalStateException.class, result::backward);
        var wrapped = new FloatWrapperAutogradValueImpl(2f, false);
        var wrappedResult = wrapped.neg();
        assertFalse(wrappedResult.requiresGrad());
        assertEquals(-2f, wrappedResult.data().get().floatValue());
        wrappedResult.gradNode().accumulate(new FloatWrapperAutogradValueImpl(1f, false));
        wrappedResult.valueNode().backward();
        assertTrue(wrapped.gradNode().gradient().isEmpty());
    }

    @Test void unaryHelperMapsContextOnceAndDispatchesOriginalOperandAndSeed() {
        var x = new FloatAutogradValue(2f, true);
        var seed = new FloatAutogradValue(4f, false);
        var contribution = new FloatAutogradValue(7f, false);
        var maps = new AtomicInteger();
        var rules = new AtomicInteger();
        var result = x.applyUnaryOperator(value -> -value, (incoming, operand) -> {
            assertSame(seed, incoming);
            assertSame(x, operand);
            rules.incrementAndGet();
            return contribution;
        }, context -> { assertNull(context); maps.incrementAndGet(); return null; });
        assertEquals(1, maps.get());
        result.valueNode().backward(); // An absent incoming gradient must skip the rule.
        assertEquals(0, rules.get());
        result.gradNode().accumulate(seed);
        result.valueNode().backward();
        assertEquals(1, rules.get());
        assertSame(contribution, x.gradNode().gradient().orElseThrow());
        assertEquals(-2f, result.data().get());
        assertEquals(1, maps.get());
    }

    @Test void unaryHelperRejectsNullArgumentsAndNullContributions() {
        var x = new FloatAutogradValue(2f, true);
        assertThrows(NullPointerException.class,
                () -> x.applyUnaryOperator(null, (incoming, operand) -> incoming, context -> context));
        assertThrows(NullPointerException.class,
                () -> x.applyUnaryOperator(value -> value, null, context -> context));
        assertThrows(NullPointerException.class,
                () -> x.applyUnaryOperator(value -> value, (incoming, operand) -> incoming, null));
        var result = x.applyUnaryOperator(value -> value, (incoming, operand) -> null, context -> context);
        assertThrows(NullPointerException.class, result::backward);
        assertTrue(x.gradNode().gradient().isEmpty());
    }
}
