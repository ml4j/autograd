package org.ml4j.autograd.scalar;

import org.ml4j.autograd.scalar.impl.FloatAutogradValueFactoryImpl;


import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class AdditionBackwardTest {
    private final FloatAutogradValueFactory rawFactory =
            new FloatAutogradValueFactoryImpl();

    private FloatAutogradValue value(float data, boolean requiresGrad) {
        return rawFactory.create(data, requiresGrad);
    }

    private float gradient(FloatAutogradValue value) {
        return value.gradNode().gradient().orElseThrow().data().get();
    }

    @Test
    void localCallbackPassesExplicitSeedToBothOperandsWithoutChangingForwardState() {
        var left = value(2f, true);
        var right = value(3f, true);
        var sum = left.add(right);
        var seed = value(4f, false);
        sum.gradNode().accumulate(seed);
        sum.valueNode().backward();
        assertSame(seed, left.gradNode().gradient().orElseThrow());
        assertSame(seed, right.gradNode().gradient().orElseThrow());
        assertSame(seed, sum.gradNode().gradient().orElseThrow());
        assertEquals(2f, left.data().get());
        assertEquals(3f, right.data().get());
        assertEquals(5f, sum.data().get());
        assertNull(left.valueNode().backwardFunction());
        assertNull(right.valueNode().backwardFunction());
    }

    @Test
    void absentIncomingGradientDoesNotPropagate() {
        var left = value(2f, true);
        var right = value(3f, true);
        left.add(right).valueNode().backward();
        assertTrue(left.gradNode().gradient().isEmpty());
        assertTrue(right.gradNode().gradient().isEmpty());
    }

    @Test
    void onlyTrackedOperandsReceiveContributions() {
        for (boolean leftTracked : new boolean[]{false, true}) {
            for (boolean rightTracked : new boolean[]{false, true}) {
                var left = value(2f, leftTracked);
                var right = value(3f, rightTracked);
                var sum = left.add(right);
                sum.gradNode().accumulate(value(1f, false));
                sum.valueNode().backward();
                assertEquals(leftTracked, left.gradNode().gradient().isPresent());
                assertEquals(rightTracked, right.gradNode().gradient().isPresent());
                if (leftTracked) assertEquals(1f, gradient(left));
                if (rightTracked) assertEquals(1f, gradient(right));
            }
        }
    }

    @Test
    void repeatedOperandReceivesTwoContributionsWithoutMutatingSeed() {
        var x = value(2f, true);
        var sum = x.add(x);
        var seed = value(3f, false);
        sum.gradNode().accumulate(seed);
        sum.valueNode().backward();
        assertEquals(6f, gradient(x));
        assertEquals(3f, seed.data().get());
        assertEquals(3f, gradient(sum));
    }

    @Test
    void contributionAddsToExistingGradientWithoutChangingEarlierExpression() {
        var x = value(2f, true);
        var earlier = value(5f, false);
        x.gradNode().accumulate(earlier);
        var sum = x.add(value(3f, false));
        sum.gradNode().accumulate(value(2f, false));
        sum.valueNode().backward();
        assertEquals(7f, gradient(x));
        assertEquals(5f, earlier.data().get());
    }

    @Test
    void localPropagationDoesNotTraverseIntermediateInputs() {
        var x = value(2f, true);
        var intermediate = x.add(value(3f, false));
        var result = intermediate.add(value(4f, false));
        result.gradNode().accumulate(value(1f, false));
        result.valueNode().backward();
        assertEquals(1f, gradient(intermediate));
        assertTrue(x.gradNode().gradient().isEmpty());
        intermediate.valueNode().backward();
        assertEquals(1f, gradient(x));
    }
}
