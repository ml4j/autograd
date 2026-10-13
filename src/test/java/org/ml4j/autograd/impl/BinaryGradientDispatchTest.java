package org.ml4j.autograd.impl;

import org.junit.jupiter.api.Test;
import org.ml4j.autograd.node.ValueNode;
import org.ml4j.autograd.operators.BinaryGradient;

import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Supplier;

import static org.junit.jupiter.api.Assertions.*;

class BinaryGradientDispatchTest {
    private static final class TestValue extends AutogradValueImpl<TestValue, Float, Void> {
        TestValue(float data, boolean requiresGrad) { this(() -> data, requiresGrad, List.of()); }
        TestValue(Supplier<Float> data, boolean requiresGrad, List<? extends ValueNode<?>> inputs) {
            super(data, null, requiresGrad, inputs);
        }
        TestValue operation(TestValue other, BinaryGradient<TestValue> left, BinaryGradient<TestValue> right) {
            return applyBinaryOperator(other, (leftData, rightData) -> leftData + rightData, left, right, this::mapContext);
        }
        @Override public TestValue add(TestValue other) {
            return applyBinaryOperator(other, (leftData, rightData) -> leftData + rightData,
                    (incoming, left, right) -> incoming, (incoming, left, right) -> incoming, this::mapContext);
        }
        @Override protected TestValue self() { return this; }
        @Override public Void mapContext(Void left, Void right) { return null; }
        @Override protected TestValue createAutogradValue(Supplier<Float> data, Void context,
                boolean requiresGrad, List<? extends ValueNode<?>> inputs) {
            return new TestValue(data, requiresGrad, inputs);
        }
    }

    @Test void dispatchesDistinctRulesWithIncomingGradientAndOriginalOperandOrder() {
        var left = new TestValue(2f, true);
        var right = new TestValue(3f, true);
        var seed = new TestValue(4f, false);
        var calls = new AtomicInteger();
        var leftContribution = new TestValue(7f, false);
        var rightContribution = new TestValue(9f, false);
        var result = left.operation(right, (incoming, first, second) -> {
            assertSame(seed, incoming); assertSame(left, first); assertSame(right, second);
            calls.incrementAndGet(); return leftContribution;
        }, (incoming, first, second) -> {
            assertSame(seed, incoming); assertSame(left, first); assertSame(right, second);
            calls.incrementAndGet(); return rightContribution;
        });
        assertEquals(0, calls.get());
        result.valueNode().backward();
        assertEquals(0, calls.get());
        result.backward(seed);
        assertEquals(2, calls.get());
        assertSame(leftContribution, left.gradNode().gradient().orElseThrow());
        assertSame(rightContribution, right.gradNode().gradient().orElseThrow());
    }

    @Test void doesNotInvokeConstantOperandsRule() {
        var constant = new TestValue(2f, false);
        var tracked = new TestValue(3f, true);
        var seed = new TestValue(1f, false);
        var result = constant.operation(tracked,
                (incoming, left, right) -> { throw new AssertionError("Constant rule must be skipped"); },
                (incoming, left, right) -> incoming);
        result.backward(seed);
        assertTrue(constant.gradNode().gradient().isEmpty());
        assertSame(seed, tracked.gradNode().gradient().orElseThrow());
    }
}
