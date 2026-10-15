package org.ml4j.autograd.scalar.impl;

import org.ml4j.autograd.scalar.FloatWrapperAutogradValueFactory;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class FloatScalarOperationsTest {
    private final FloatWrapperAutogradValueFactory factory =
            new FloatWrapperAutogradValueFactoryImpl();

    @Test
    void scalarArithmeticReturnsNewWrappersAndPreservesInput() {
        var input = new FloatWrapperImpl(6f);
        assertEquals(8f, input.add(2f).floatValue());
        assertEquals(4f, input.sub(2f).floatValue());
        assertEquals(12f, input.mul(2f).floatValue());
        assertEquals(3f, input.div(2f).floatValue());
        assertNotSame(input, input.add(0f));
        assertEquals(6f, input.floatValue());
        assertEquals(Float.POSITIVE_INFINITY, input.div(0f).floatValue());
        assertEquals(Float.NEGATIVE_INFINITY, input.div(-0f).floatValue());
        assertTrue(Float.isNaN(new FloatWrapperImpl(0f).div(0f).floatValue()));
    }

    @Test
    void masksRespectEqualityNaNInfinitiesAndSignedZero() {
        var input = new FloatWrapperImpl(2f);
        assertEquals(0f, input.gt(2f).floatValue());
        assertEquals(1f, input.gte(2f).floatValue());
        assertEquals(1f, input.gt(1f).floatValue());
        assertEquals(0f, input.gte(3f).floatValue());
        for (float number : new float[]{Float.NaN, -0f, 0f, Float.NEGATIVE_INFINITY, Float.POSITIVE_INFINITY}) {
            var value = new FloatWrapperImpl(number);
            assertEquals(number > 0f ? 1f : 0f, value.gt(0f).floatValue());
            assertEquals(number >= 0f ? 1f : 0f, value.gte(0f).floatValue());
            assertEquals(0f, value.gt(Float.NaN).floatValue());
            assertEquals(0f, value.gte(Float.NaN).floatValue());
        }
    }

    @Test
    void arrayAccessPreservesScalarAndWrappedGradientState() {
        var numerical = new FloatWrapperImpl(-0f);
        var array = numerical.dataAsFloatArray();
        assertEquals(Float.floatToRawIntBits(-0f), Float.floatToRawIntBits(array[0]));
        assertNotSame(array, numerical.dataAsFloatArray());
        var value = factory.create(() -> numerical, null, true);
        assertArrayEquals(array, value.dataAsFloatArray());
        assertTrue(value.gradNode().gradient().isEmpty());
        assertTrue(value.requiresGrad());
    }
}
