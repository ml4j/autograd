package org.ml4j.autograd.impl;

import org.ml4j.autograd.scalar.FloatAutogradValueFactory;

import org.ml4j.autograd.scalar.impl.FloatAutogradValueFactoryImpl;


import org.junit.jupiter.api.Test;
import org.ml4j.autograd.Parameter;
import org.ml4j.autograd.scalar.FloatAutogradValue;

import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.*;

class ParameterImplTest {
    private final FloatAutogradValueFactory rawFactory =
            new FloatAutogradValueFactoryImpl();

    private record Tensor(FloatAutogradValue autogradValue) {}

    @Test
    void wrapperExtractionIsLazyAndOldGraphsRetainTheirLeaves() {
        var reads = new AtomicInteger();
        var extractions = new AtomicInteger();
        var original = new Tensor(rawFactory.create(() -> {
            reads.incrementAndGet();
            return 2f;
        }, null, true));
        var replacement = new Tensor(rawFactory.create(3f, true));
        Parameter<Tensor> parameter = new ParameterImpl<>(original, tensor -> {
            extractions.incrementAndGet();
            return tensor.autogradValue();
        });
        assertSame(original, parameter.value());
        assertSame(original, parameter.value());
        var oldGraph = parameter.value().autogradValue().mul(original.autogradValue());
        parameter.replace(replacement);
        assertSame(replacement, parameter.value());
        assertEquals(2, extractions.get());
        assertEquals(0, reads.get());
        oldGraph.backward();
        assertEquals(4f, original.autogradValue().gradNode().gradient().orElseThrow().data().get());
        assertTrue(replacement.autogradValue().gradNode().gradient().isEmpty());
        var newGraph = replacement.autogradValue().mul(replacement.autogradValue());
        newGraph.backward();
        assertEquals(6f, replacement.autogradValue().gradNode().gradient().orElseThrow().data().get());
    }

    @Test
    void invalidExtractionsPreserveTheCurrentWrapper() {
        var original = new Tensor(rawFactory.create(2f, true));
        var failure = new IllegalStateException("extraction failed");
        var throwing = new Tensor(rawFactory.create(3f, true));
        var parameter = new ParameterImpl<>(original, tensor -> {
            if (tensor == throwing) throw failure;
            return tensor.autogradValue();
        });
        var nonLeaf = new Tensor(original.autogradValue().neg());
        assertThrows(IllegalArgumentException.class, () -> new ParameterImpl<>(nonLeaf, Tensor::autogradValue));
        assertThrows(NullPointerException.class, () -> new ParameterImpl<>(original, null));
        assertThrows(NullPointerException.class, () -> new ParameterImpl<Tensor>(null, Tensor::autogradValue));
        assertThrows(NullPointerException.class, () -> new ParameterImpl<>(new Tensor(null), Tensor::autogradValue));
        assertThrows(IllegalArgumentException.class, () -> parameter.replace(nonLeaf));
        assertSame(original, parameter.value());
        assertThrows(NullPointerException.class, () -> parameter.replace(new Tensor(null)));
        assertSame(original, parameter.value());
        assertThrows(NullPointerException.class, () -> parameter.replace(null));
        assertSame(original, parameter.value());
        assertSame(failure, assertThrows(IllegalStateException.class, () -> parameter.replace(throwing)));
        assertSame(original, parameter.value());
    }

    @Test
    void directConstructorRejectsNonAutogradValuesAtRuntime() {
        assertThrows(IllegalArgumentException.class, () -> new ParameterImpl<>("wrapper"));
        Parameter<Object> parameter = new ParameterImpl<>(rawFactory.create(2f, true));
        var original = parameter.value();
        assertThrows(IllegalArgumentException.class, () -> parameter.replace("wrapper"));
        assertSame(original, parameter.value());
    }

    @Test
    void constructionAndReplacementAreLazyAndPreserveLeafState() {
        var reads = new AtomicInteger();
        var original = rawFactory.create(() -> { reads.incrementAndGet(); return 2f; }, null, true);
        var replacement = rawFactory.create(() -> { reads.incrementAndGet(); return 3f; }, null, false);
        var originalGradient = rawFactory.create(5f, false);
        var replacementGradient = rawFactory.create(7f, false);
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
        var leaf = rawFactory.create(() -> { reads.incrementAndGet(); return 2f; }, null, true);
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
        var original = rawFactory.create(2f, true);
        var parameter = new ParameterImpl<>(original);
        var oldValue = parameter.value();
        var oldGraph = oldValue.mul(oldValue);
        var replacement = rawFactory.create(3f, true);
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
