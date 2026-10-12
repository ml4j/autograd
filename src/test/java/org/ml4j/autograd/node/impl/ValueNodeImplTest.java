package org.ml4j.autograd.node.impl;

import org.junit.jupiter.api.Test;
import org.ml4j.autograd.Value;
import org.ml4j.autograd.node.ValueNode;
import org.ml4j.autograd.scalar.FloatValue;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;
import java.util.function.Supplier;

import static org.junit.jupiter.api.Assertions.*;

class ValueNodeImplTest {
    @Test
    void constructionAndMetadataDoNotResolveValue() {
        var resolutions = new AtomicInteger();
        var value = new FloatValue(2f);
        var node = new ValueNodeImpl<>(() -> { resolutions.incrementAndGet(); return value; });
        assertTrue(node.inputs().isEmpty());
        assertNull(node.backwardFunction());
        node.backward();
        assertEquals(0, resolutions.get());
        assertSame(value, node.value());
        assertEquals(1, resolutions.get());
    }

    @Test
    void valueAccessDoesNotEvaluateNumericalData() {
        var evaluations = new AtomicInteger();
        Value<Float, Void> value = new Value<>() {
            @Override public Supplier<Float> data() {
                return () -> { evaluations.incrementAndGet(); return 2f; };
            }
            @Override public Void context() { return null; }
        };
        var node = new ValueNodeImpl<>(() -> value);
        assertSame(value, node.value());
        node.setBackwardFunction(resultValue -> assertSame(value, resultValue));
        node.backward();
        assertEquals(0, evaluations.get());
    }

    @Test
    void inputsAreCopiedImmutablyWithOrderAndDuplicatesPreserved() {
        var left = new ValueNodeImpl<>(() -> new FloatValue(2f));
        var right = new ValueNodeImpl<>(() -> new FloatValue(3f));
        var supplied = new ArrayList<ValueNode<?>>(List.of(left, right, left));
        var result = new ValueNodeImpl<>(() -> new FloatValue(7f), supplied);
        supplied.clear();
        assertEquals(List.of(left, right, left), result.inputs());
        assertThrows(UnsupportedOperationException.class, () -> result.inputs().clear());
        assertThrows(UnsupportedOperationException.class, () -> left.inputs().add(null));
    }

    @Test
    void callbackCanBeInstalledReplacedAndRemoved() {
        var value = new FloatValue(5f);
        var node = new ValueNodeImpl<>(() -> value);
        var firstCalls = new AtomicInteger();
        var secondCalls = new AtomicInteger();
        Consumer<FloatValue> first = resultValue -> {
            assertSame(value, resultValue);
            firstCalls.incrementAndGet();
        };
        Consumer<FloatValue> second = resultValue -> secondCalls.incrementAndGet();
        node.setBackwardFunction(first);
        assertSame(first, node.backwardFunction());
        assertEquals(0, firstCalls.get());
        node.backward();
        assertEquals(1, firstCalls.get());
        node.setBackwardFunction(second);
        assertSame(second, node.backwardFunction());
        assertEquals(0, secondCalls.get());
        node.backward();
        assertEquals(1, firstCalls.get());
        assertEquals(1, secondCalls.get());
        node.setBackwardFunction(null);
        assertNull(node.backwardFunction());
        node.backward();
        assertEquals(1, secondCalls.get());
    }

    @Test
    void backwardDoesNotTraverseInputs() {
        var calls = new AtomicInteger();
        var input = new ValueNodeImpl<>(() -> new FloatValue(2f));
        input.setBackwardFunction(value -> calls.incrementAndGet());
        var result = new ValueNodeImpl<>(() -> new FloatValue(4f), List.of(input, input));
        result.setBackwardFunction(value -> {});
        result.backward();
        assertEquals(0, calls.get());
    }

    @Test
    void callbackFailurePropagatesWithoutRollback() {
        var node = new ValueNodeImpl<>(() -> new FloatValue(2f));
        var calls = new AtomicInteger();
        var failure = new IllegalStateException("callback failed");
        node.setBackwardFunction(value -> { calls.incrementAndGet(); throw failure; });
        assertSame(failure, assertThrows(IllegalStateException.class, node::backward));
        assertEquals(1, calls.get());
    }

    @Test
    void rejectsNullConstructionArgumentsAndNullRepresentedValue() {
        assertThrows(NullPointerException.class, () -> new ValueNodeImpl<FloatValue>(null));
        assertThrows(NullPointerException.class, () -> new ValueNodeImpl<>(() -> new FloatValue(2f), null));
        var inputs = new ArrayList<ValueNode<?>>();
        inputs.add(null);
        assertThrows(NullPointerException.class, () -> new ValueNodeImpl<>(() -> new FloatValue(2f), inputs));
        var node = new ValueNodeImpl<FloatValue>(() -> null);
        assertThrows(NullPointerException.class, node::value);
    }
}
