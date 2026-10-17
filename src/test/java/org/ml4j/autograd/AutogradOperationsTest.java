package org.ml4j.autograd;

import org.junit.jupiter.api.Test;
import org.ml4j.autograd.scalar.FloatAutogradValue;
import org.ml4j.autograd.scalar.impl.FloatAutogradValueFactoryImpl;

import static org.junit.jupiter.api.Assertions.*;

class AutogradOperationsTest {
    @Test
    void capabilitySupportsTrackingAndBothBackwardEntryPoints() {
        var factory = new FloatAutogradValueFactoryImpl();
        var input = factory.create(3f, false);
        AutogradOperations<FloatAutogradValue> tracking = input;
        assertFalse(tracking.requiresGrad());
        tracking.setRequiresGrad(true);
        assertTrue(tracking.requiresGrad());

        var result = input.mul(input);
        AutogradOperations<FloatAutogradValue> operations = result;
        assertThrows(IllegalStateException.class, () -> operations.setRequiresGrad(false));
        assertThrows(NullPointerException.class, () -> operations.backward(null));
        operations.backward();
        assertEquals(6f, input.gradNode().gradient().orElseThrow().data().get().floatValue());
        operations.backward(factory.create(2f, false));
        assertEquals(18f, input.gradNode().gradient().orElseThrow().data().get().floatValue());
    }
}
