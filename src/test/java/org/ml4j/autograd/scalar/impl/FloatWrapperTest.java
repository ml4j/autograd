package org.ml4j.autograd.scalar.impl;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class FloatWrapperTest {
    @Test void numericalArithmeticReturnsCorrectResultsWithoutMutatingOperands() {
        var left = new FloatWrapperImpl(2f);
        var right = new FloatWrapperImpl(3f);
        assertEquals(5f, left.add(right).floatValue());
        assertEquals(6f, left.mul(right).floatValue());
        assertEquals(2f, left.floatValue());
        assertEquals(3f, right.floatValue());
    }
}
