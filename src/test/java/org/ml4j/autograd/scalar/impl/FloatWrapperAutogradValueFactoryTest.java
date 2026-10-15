package org.ml4j.autograd.scalar.impl;

import org.ml4j.autograd.scalar.FloatWrapper;
import org.ml4j.autograd.scalar.FloatWrapperAutogradValue;
import org.ml4j.autograd.scalar.FloatWrapperAutogradValueFactory;

import org.junit.jupiter.api.Test;
import org.ml4j.autograd.AutogradValueFactory;
import java.util.concurrent.atomic.AtomicInteger;
import static org.junit.jupiter.api.Assertions.*;

class FloatWrapperAutogradValueFactoryTest {
    private final FloatWrapperAutogradValueFactory factory = new FloatWrapperAutogradValueFactoryImpl();

    @Test
    void primitiveCreationProducesIndependentLeavesWithExplicitTracking() {
        for (boolean tracked : new boolean[]{false, true}) {
            var first = factory.create(3f, tracked);
            var second = factory.create(3f, tracked);
            assertNotSame(first, second);
            assertNotSame(first.valueNode(), second.valueNode());
            assertNotSame(first.gradNode(), second.gradNode());
            assertNull(first.context());
            assertEquals(tracked, first.requiresGrad());
            assertTrue(first.valueNode().inputs().isEmpty());
            assertTrue(first.gradNode().gradient().isEmpty());
            first.valueNode().backward();
            assertTrue(first.gradNode().gradient().isEmpty());
            assertEquals(3f, first.data().get().floatValue());
            first.gradNode().accumulate(factory.create(7f, false));
            assertTrue(second.gradNode().gradient().isEmpty());
        }
    }

    @Test
    void genericSupplierCreationPreservesLazyNonCachingConstructorSemantics() {
        AutogradValueFactory<FloatWrapperAutogradValue, FloatWrapper, Void> generic = factory;
        var reads = new AtomicInteger();
        FloatWrapper data = new FloatWrapperImpl(2f);
        var leaf = generic.create(() -> { reads.incrementAndGet(); return data; }, null, true);
        assertEquals(0, reads.get());
        assertTrue(leaf.requiresGrad());
        assertNull(leaf.context());
        assertTrue(leaf.valueNode().inputs().isEmpty());
        assertTrue(leaf.gradNode().gradient().isEmpty());
        assertSame(data, leaf.data().get());
        assertSame(data, leaf.data().get());
        assertEquals(2, reads.get());
        assertThrows(NullPointerException.class, () -> generic.create(null, null, false));
    }

    @Test
    void factoryLeavesParticipateInRepeatedBackwardAndHigherDerivatives() {
        var input = factory.create(3f, true);
        var constant = factory.create(2f, false);
        var result = input.mul(input).mul(constant);
        result.backward();
        var first = input.gradNode().gradient().orElseThrow();
        assertEquals(12f, first.data().get().floatValue());
        assertTrue(constant.gradNode().gradient().isEmpty());
        result.backward();
        assertEquals(24f, input.gradNode().gradient().orElseThrow().data().get().floatValue());
        input.gradNode().reset();
        first.backward();
        assertEquals(4f, input.gradNode().gradient().orElseThrow().data().get().floatValue());
    }
}
