import numpy as np
import unittest
from scipy.sparse import csr_matrix

class TestCsrMatrixRound(unittest.TestCase):
    def test_round_raises_before_patch(self):
        """Ensure round raises TypeError before __round__ is defined"""

        operation = csr_matrix([[1.111, 0], [0, 2.222]])
        try:
            round(operation)

        except TypeError as e:
            self.assertIn("__round__", str(e))
        else:
            self.fail("TypeError not raised when calling round on csr_matrix before patch")

    def test_round_functionality_after_patch(self):
        """Test that round(csr_matrix) return rounded values correctly"""

        operation = csr_matrix([[1.111, 0.0], [0.0, 2.999]])
        rounded = round(operation)
        expected = csr_matrix([[1.0, 0.0], [0.0, 3.0]])
        self.assertTrue((rounded != expected).nnz == 0)
        self.assertTrue(np.allclose(rounded.data, expected.data))

if __name__ == "__main__":
    unittest.main()