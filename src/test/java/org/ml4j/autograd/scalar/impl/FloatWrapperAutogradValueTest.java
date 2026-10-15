package org.ml4j.autograd.scalar.impl;

import org.ml4j.autograd.scalar.FloatWrapperAutogradValueFactory;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class FloatWrapperAutogradValueTest {
    private final FloatWrapperAutogradValueFactory factory =
            new FloatWrapperAutogradValueFactoryImpl();

    @Test void defaultAdditionHonoursBothOperandsGradientRequirements() {
        var left = factory.create(2f, false);
        var right = factory.create(3f, true);
        var sum = left.add(right);
        assertEquals(5f, sum.data().get().floatValue());
        assertTrue(sum.requiresGrad());
        sum.backward();
        assertTrue(left.gradNode().gradient().isEmpty());
        assertEquals(1f, right.gradNode().gradient().orElseThrow().data().get().floatValue());
    }

    @Test void multiplicationMatchesForwardAndBackwardRules() {
        var left = factory.create(2f, true);
        var right = factory.create(3f, true);
        var product = left.mul(right);
        assertEquals(6f, product.data().get().floatValue());
        product.backward(factory.create(4f, false));
        assertEquals(12f, left.gradNode().gradient().orElseThrow().data().get().floatValue());
        assertEquals(8f, right.gradNode().gradient().orElseThrow().data().get().floatValue());
    }

    @Test void wrapperArithmeticRetainsHigherDerivativeGraphs() {
        var x = factory.create(0.5f, true);
        var y = factory.create(0.6f, true);
        x.mul(x).add(y.mul(x)).add(y.mul(y)).backward();
        var gx = x.gradNode().gradient().orElseThrow();
        var gy = y.gradNode().gradient().orElseThrow();
        assertEquals(1.6f, gx.data().get().floatValue(), 0.0001f);
        assertEquals(1.7f, gy.data().get().floatValue(), 0.0001f);
        x.gradNode().reset(); y.gradNode().reset();
        gx.mul(factory.create(2f, false)).add(gy).backward();
        assertEquals(5f, x.gradNode().gradient().orElseThrow().data().get().floatValue(), 0.0001f);
        assertEquals(4f, y.gradNode().gradient().orElseThrow().data().get().floatValue(), 0.0001f);
    }

    @Test void inheritedProductDefaultHandlesRepeatedOperandsAndRepeatedPasses() {
        var x = factory.create(3f, true);
        var square = x.mul(x);
        square.backward();
        var first = x.gradNode().gradient().orElseThrow();
        assertEquals(6f, first.data().get().floatValue());
        square.backward();
        assertEquals(12f, x.gradNode().gradient().orElseThrow().data().get().floatValue());
        assertEquals(6f, first.data().get().floatValue());
        assertEquals(3f, x.data().get().floatValue());
    }
}
