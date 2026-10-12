package org.ml4j.autograd.scalar;

import org.junit.jupiter.api.Test;
import org.ml4j.autograd.Value;

import static org.junit.jupiter.api.Assertions.*;

class FloatValueTest {

    @Test
    void suppliesStableNonNullDataAcrossRepeatedReads() {
        Value<Float, Void> value = new FloatValue(2.5f);
        var supplier = value.data();
        assertNotNull(supplier);
        assertNotNull(supplier.get());
        assertEquals(2.5f, supplier.get());
        assertEquals(2.5f, supplier.get());
        assertEquals(2.5f, value.data().get());
    }

    @Test
    void scalarNeedsNoContext() {
        var value = new FloatValue(3f);
        assertNull(value.context());
        assertNull(value.context());
        assertEquals(3f, value.data().get());
    }

    @Test
    void separateValuesRetainTheirOwnData() {
        var first = new FloatValue(2f);
        var second = new FloatValue(-3f);
        assertEquals(2f, first.data().get());
        assertEquals(-3f, second.data().get());
        assertEquals(2f, first.data().get());
    }

    @Test
    void preservesFloatingPointBoundaryValues() {
        float[] values = {0f, -0f, Float.MIN_VALUE, Float.MAX_VALUE,
                Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY, Float.NaN};
        for (float expected : values) {
            var value = new FloatValue(expected);
            assertEquals(Float.floatToIntBits(expected), Float.floatToIntBits(value.data().get()));
        }
    }
}
