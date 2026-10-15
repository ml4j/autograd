package org.ml4j.autograd.scalar;

import org.ml4j.autograd.scalar.impl.FloatAutogradValueFactoryImpl;


import org.junit.jupiter.api.Test;
import org.ml4j.autograd.impl.AutogradValueImpl;
import org.ml4j.autograd.node.ValueNode;

import java.util.List;
import java.util.function.Consumer;
import java.util.function.Supplier;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.*;

class BackwardPassTest {
    private final FloatAutogradValueFactory rawFactory =
            new FloatAutogradValueFactoryImpl();

    private FloatAutogradValue tracked(float value) { return rawFactory.create(value, true); }
    private FloatAutogradValue seed(float value) { return rawFactory.create(value, false); }
    private float gradient(FloatAutogradValue value) {
        return value.gradNode().gradient().orElseThrow().data().get();
    }

    @Test void explicitAndDefaultSeedsRespectConstants() {
        var x = tracked(2); var constant = seed(3); var sum = constant.add(x);
        sum.backward(seed(4));
        assertEquals(4f, gradient(x));
        assertTrue(constant.gradNode().gradient().isEmpty());
        x.gradNode().reset();
        sum.backward();
        assertEquals(1f, gradient(x));
        assertEquals(5f, sum.data().get());
    }

    @Test void sharedIntermediateRunsOnceAfterBothContributionsArrive() {
        var x = tracked(2); var y = tracked(3);
        var shared = x.add(y);
        var output = shared.add(shared);
        var calls = new AtomicInteger();
        var original = shared.valueNode().backwardFunction();
        shared.valueNode().setBackwardFunction(result -> {
            calls.incrementAndGet();
            assertEquals(2f, gradient(result));
            original.accept(result);
        });
        output.backward();
        assertEquals(1, calls.get());
        assertEquals(2f, gradient(x));
        assertEquals(2f, gradient(y));
    }

    @Test void repeatedPassesAccumulateLeavesAndRefreshIntermediates() {
        var x = tracked(2); var shared = x.add(x); var output = shared.add(shared);
        var handle = x.gradNode();
        output.backward();
        var earlier = handle.gradient().orElseThrow();
        assertEquals(4f, gradient(x));
        output.backward(seed(2));
        assertEquals(12f, gradient(x));
        assertEquals(4f, gradient(shared));
        assertEquals(2f, gradient(output));
        assertEquals(4f, earlier.data().get());
        handle.reset();
        output.backward();
        assertSame(handle, x.gradNode());
        assertEquals(4f, gradient(x));
    }

    @Test void leafEntryPointAccumulatesOnlyNewSeeds() {
        var x = tracked(2);
        x.backward(); x.backward(seed(3));
        assertEquals(4f, gradient(x));
    }

    @Test void invalidSeedAndRootPreserveExistingGradient() {
        var x = tracked(2); x.backward();
        var earlier = x.gradNode().gradient().orElseThrow();
        assertThrows(NullPointerException.class, () -> x.backward(null));
        assertSame(earlier, x.gradNode().gradient().orElseThrow());
        var constant = seed(3);
        assertThrows(IllegalStateException.class, constant::backward);
        assertThrows(IllegalStateException.class, () -> constant.backward(seed(1)));
        assertTrue(constant.gradNode().gradient().isEmpty());
    }

    // A test-only numerical type uses the base constructor to exercise malformed graph rejection.
    private static final class GraphFixture extends AutogradValueImpl<GraphFixture, Float, Void> {
        GraphFixture(List<? extends ValueNode<?>> inputs) { super(() -> 2f, null, true, inputs); }
        @Override public GraphFixture add(GraphFixture other) {
            return applyBinaryOperator(other, (leftData, rightData) -> leftData + rightData,
                    (incoming, left, right) -> incoming, (incoming, left, right) -> incoming, this::mapContext);
        }
        @Override protected GraphFixture self() { return this; }
        @Override public Void mapContext(Void left, Void right) { return null; }
        @Override protected GraphFixture createAutogradValue(Supplier<Float> data, Void context,
                boolean requiresGrad, List<? extends ValueNode<?>> inputs) {
            throw new UnsupportedOperationException("No operations required by graph validation tests");
        }
    }

    @Test void cyclicGraphIsRejectedBeforeGradientStateChanges() {
        var owner = new AtomicReference<GraphFixture>();
        ValueNode<GraphFixture> cyclic = new ValueNode<>() {
            @Override public GraphFixture value() { return owner.get(); }
            @Override public List<? extends ValueNode<?>> inputs() { return List.of(this); }
            @Override public void setBackwardFunction(Consumer<GraphFixture> callback) {}
            @Override public Consumer<GraphFixture> backwardFunction() { return null; }
            @Override public void backward() { fail("Cycle must be rejected before execution"); }
        };
        var root = new GraphFixture(List.of(cyclic));
        owner.set(root);
        var seed = new GraphFixture(List.of());
        root.gradNode().accumulate(seed);
        assertThrows(IllegalStateException.class, () -> root.backward(seed));
        assertSame(seed, root.gradNode().gradient().orElseThrow());
    }

    @Test void numericalTypesWithoutImplicitSeedRequireExplicitSeeds() {
        var root = new GraphFixture(List.of());
        assertThrows(UnsupportedOperationException.class, root::backward);
        var seed = new GraphFixture(List.of());
        root.backward(seed);
        assertSame(seed, root.gradNode().gradient().orElseThrow());
    }
}
