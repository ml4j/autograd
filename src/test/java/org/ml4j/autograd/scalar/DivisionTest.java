package org.ml4j.autograd.scalar;

import org.ml4j.autograd.scalar.impl.FloatAutogradValueFactoryImpl;


import org.ml4j.autograd.scalar.impl.FloatWrapperAutogradValueFactoryImpl;

import org.junit.jupiter.api.Test;
import org.ml4j.autograd.scalar.impl.FloatWrapperImpl;
import java.util.concurrent.atomic.AtomicInteger;
import static org.junit.jupiter.api.Assertions.*;

class DivisionTest {
    private final FloatAutogradValueFactory rawFactory =
            new FloatAutogradValueFactoryImpl();

    private final FloatWrapperAutogradValueFactory factory =
            new FloatWrapperAutogradValueFactoryImpl();

    @Test void rawQuotientIsLazyAndExplicitSeedProducesBothContributions() {
        var reads = new AtomicInteger();
        var left = rawFactory.create(() -> { reads.incrementAndGet(); return 6f; }, null, true);
        var right = rawFactory.create(2f, true);
        var result = left.div(right);
        assertEquals(0, reads.get());
        assertSame(left.valueNode(), result.valueNode().inputs().getFirst());
        assertSame(right.valueNode(), result.valueNode().inputs().get(1));
        assertEquals(3f, result.data().get());
        result.backward(rawFactory.create(4f, false));
        assertEquals(2f, left.gradNode().gradient().orElseThrow().data().get());
        assertEquals(-6f, right.gradNode().gradient().orElseThrow().data().get());
        assertEquals(6f, left.data().get());
        assertEquals(2f, right.data().get());
    }

    @Test void wrappedDefaultSupportsRepeatedPassesAndMixedSecondDerivatives() {
        var left = factory.create(6f, true);
        var right = factory.create(2f, true);
        var result = left.div(right);
        assertEquals(3f, result.data().get().floatValue());
        result.backward();
        var firstRight = right.gradNode().gradient().orElseThrow();
        assertEquals(0.5f, left.gradNode().gradient().orElseThrow().data().get().floatValue());
        assertEquals(-1.5f, firstRight.data().get().floatValue());
        result.backward();
        assertEquals(1f, left.gradNode().gradient().orElseThrow().data().get().floatValue());
        assertEquals(-3f, right.gradNode().gradient().orElseThrow().data().get().floatValue());
        assertEquals(-1.5f, firstRight.data().get().floatValue());
        left.gradNode().reset(); right.gradNode().reset();
        firstRight.backward(); // d(-left/right²)/dleft=-1/4; /dright=2*left/right³=1.5.
        assertEquals(-0.25f, left.gradNode().gradient().orElseThrow().data().get().floatValue());
        assertEquals(1.5f, right.gradNode().gradient().orElseThrow().data().get().floatValue());
    }

    @Test void reciprocalRetainsRawSecondDerivativeGraph() {
        var x = rawFactory.create(2f, true);
        rawFactory.create(1f, false).div(x).backward();
        var first = x.gradNode().gradient().orElseThrow();
        assertEquals(-0.25f, first.data().get());
        x.gradNode().reset();
        first.backward();
        assertEquals(0.25f, x.gradNode().gradient().orElseThrow().data().get());
    }

    @Test void sharedOperandQuotientCancelsContributions() {
        var x = rawFactory.create(2f, true);
        var result = x.div(x);
        assertEquals(1f, result.data().get());
        result.backward();
        var first = x.gradNode().gradient().orElseThrow();
        assertEquals(0f, first.data().get());
        x.gradNode().reset();
        first.backward();
        assertEquals(0f, x.gradNode().gradient().orElseThrow().data().get(), 0.00001f);
    }

    @Test void constantsAreSkippedInBothPositions() {
        var tracked = factory.create(2f, true);
        var constant = factory.create(6f, false);
        constant.div(tracked).backward();
        assertEquals(-1.5f, tracked.gradNode().gradient().orElseThrow().data().get().floatValue());
        assertTrue(constant.gradNode().gradient().isEmpty());
        tracked.gradNode().reset();
        tracked.div(constant).backward();
        assertEquals(1f / 6f, tracked.gradNode().gradient().orElseThrow().data().get().floatValue());
        assertTrue(constant.gradNode().gradient().isEmpty());
        assertFalse(constant.div(factory.create(2f, false)).requiresGrad());
    }

    @Test void floatingPointBoundariesAndNullsFollowNumericalPolicy() {
        var one = new FloatWrapperImpl(1f);
        assertEquals(Float.POSITIVE_INFINITY, one.div(new FloatWrapperImpl(0f)).floatValue());
        assertEquals(Float.NEGATIVE_INFINITY, one.div(new FloatWrapperImpl(-0f)).floatValue());
        assertTrue(Float.isNaN(new FloatWrapperImpl(0f).div(new FloatWrapperImpl(0f)).floatValue()));
        assertTrue(Float.isNaN(one.div(new FloatWrapperImpl(Float.NaN)).floatValue()));
        assertEquals(0f, one.div(new FloatWrapperImpl(Float.POSITIVE_INFINITY)).floatValue());
        var raw = rawFactory.create(1f, true);
        var singular = raw.div(rawFactory.create(0f, false));
        assertEquals(Float.POSITIVE_INFINITY, singular.data().get());
        singular.backward();
        assertEquals(Float.POSITIVE_INFINITY, raw.gradNode().gradient().orElseThrow().data().get());
        var wrapped = factory.create(1f, true);
        assertThrows(NullPointerException.class, () -> one.div(null));
        assertThrows(NullPointerException.class, () -> raw.div(null));
        assertThrows(NullPointerException.class, () -> wrapped.div(null));
    }
}
