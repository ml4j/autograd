package org.ml4j.autograd.impl;

import org.junit.jupiter.api.Test;
import org.ml4j.autograd.Parameter;
import org.ml4j.autograd.scalar.FloatAutogradValue;

import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.*;

class ParameterImplTest {
    @Test
    void constructionAndReplacementAreLazyAndPreserveLeafState() {
        var reads = new AtomicInteger();
        var original = new FloatAutogradValue(() -> { reads.incrementAndGet(); return 2f; }, true);
        var replacement = new FloatAutogradValue(() -> { reads.incrementAndGet(); return 3f; }, false);
        var originalGradient = new FloatAutogradValue(5f, false);
        var replacementGradient = new FloatAutogradValue(7f, false);
        original.gradNode().accumulate(originalGradient);
        replacement.gradNode().accumulate(replacementGradient);
        var originalNode = original.valueNode();
        var replacementNode = replacement.valueNode();
        Parameter<FloatAutogradValue> parameter = new ParameterImpl<>(original);
        assertSame(original, parameter.value());
        assertSame(original, parameter.value());
        parameter.replace(replacement);
        assertSame(replacement, parameter.value());
        parameter.replace(replacement);
        assertSame(replacement, parameter.value());
        assertEquals(0, reads.get());
        assertSame(originalNode, original.valueNode());
        assertSame(replacementNode, replacement.valueNode());
        assertSame(originalGradient, original.gradNode().gradient().orElseThrow());
        assertSame(replacementGradient, replacement.gradNode().gradient().orElseThrow());
        assertTrue(original.requiresGrad());
        assertFalse(replacement.requiresGrad());
        assertEquals(2f, original.data().get());
        assertEquals(3f, parameter.value().data().get());
    }

    @Test
    void invalidInitialAndReplacementValuesAreRejectedWithoutChangingTheHandle() {
        var reads = new AtomicInteger();
        var leaf = new FloatAutogradValue(() -> { reads.incrementAndGet(); return 2f; }, true);
        var binary = leaf.add(leaf);
        var unary = leaf.neg();
        assertThrows(NullPointerException.class, () -> new ParameterImpl<FloatAutogradValue>(null));
        var parameter = new ParameterImpl<>(leaf);
        for (var result : new FloatAutogradValue[]{binary, unary}) {
            assertThrows(IllegalArgumentException.class, () -> new ParameterImpl<>(result));
            assertThrows(IllegalArgumentException.class, () -> parameter.replace(result));
            assertSame(leaf, parameter.value());
        }
        assertThrows(NullPointerException.class, () -> parameter.replace(null));
        assertSame(leaf, parameter.value());
        assertEquals(0, reads.get());
    }

    @Test
    void oldAndNewGraphsDifferentiateTheirOwnLeavesAfterReplacement() {
        var original = new FloatAutogradValue(2f, true);
        var parameter = new ParameterImpl<>(original);
        var oldValue = parameter.value();
        var oldGraph = oldValue.mul(oldValue);
        var replacement = new FloatAutogradValue(3f, true);
        parameter.replace(replacement);
        var newValue = parameter.value();
        var newGraph = newValue.mul(newValue);
        assertEquals(4f, oldGraph.data().get());
        assertEquals(9f, newGraph.data().get());
        oldGraph.backward();
        assertEquals(4f, original.gradNode().gradient().orElseThrow().data().get());
        assertTrue(replacement.gradNode().gradient().isEmpty());
        newGraph.backward();
        assertEquals(6f, replacement.gradNode().gradient().orElseThrow().data().get());
        assertEquals(4f, original.gradNode().gradient().orElseThrow().data().get());
    }
}
