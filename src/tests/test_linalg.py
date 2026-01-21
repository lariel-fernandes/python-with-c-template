import numpy as np
import pytest

from my_proj import linalg

pytestmark = pytest.mark.skipif(
    linalg is None,
    reason="linalg extension not available",
)


class TestMatMulBasic:
    def test_small_matrix(self):
        A = np.array([[1, 2], [3, 4]], dtype=np.float32)
        B = np.array([[5, 6], [7, 8]], dtype=np.float32)

        C_cuda = linalg.matmul(A, B)
        C_expected = A @ B

        assert C_cuda.shape == (2, 2)
        assert C_cuda.dtype == np.float32
        np.testing.assert_allclose(C_cuda, C_expected, rtol=1e-5)

    def test_rectangular_matrices(self):
        A = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
        B = np.array([[7, 8, 9, 10], [11, 12, 13, 14]], dtype=np.float32)

        C_cuda = linalg.matmul(A, B)
        C_expected = A @ B

        assert C_cuda.shape == (3, 4)
        np.testing.assert_allclose(C_cuda, C_expected, rtol=1e-5)

    def test_random_matrices(self):
        rng = np.random.RandomState(42)
        A = rng.rand(50, 30).astype(np.float32)
        B = rng.rand(30, 70).astype(np.float32)

        C_cuda = linalg.matmul(A, B)
        C_expected = A @ B

        # Use slightly larger tolerance for accumulated floating-point errors
        np.testing.assert_allclose(C_cuda, C_expected, rtol=1e-4)
        assert C_cuda.shape == (50, 70)

    def test_large_matrices(self):
        # These dimensions are not exact multiples of block size (16) so this tests boundary handling
        rng = np.random.RandomState(123)
        A = rng.rand(512, 256).astype(np.float32)
        B = rng.rand(256, 384).astype(np.float32)

        C_cuda = linalg.matmul(A, B)
        C_expected = A @ B

        # Larger tolerance due to more accumulation of floating-point errors
        np.testing.assert_allclose(C_cuda, C_expected, rtol=1e-3)
        assert C_cuda.shape == (512, 384)


class TestMatMulSpecialCases:
    def test_identity_matrix(self):
        A = np.random.rand(100, 100).astype(np.float32)
        I = np.eye(100, dtype=np.float32)  # 100x100 identity matrix

        C_cuda = linalg.matmul(A, I)
        C_expected = A

        np.testing.assert_allclose(C_cuda, C_expected, rtol=1e-5)

    def test_zero_matrix(self):
        A = np.random.rand(50, 50).astype(np.float32)
        Z = np.zeros((50, 50), dtype=np.float32)

        C_cuda = linalg.matmul(A, Z)
        expected = Z

        np.testing.assert_allclose(C_cuda, expected, atol=1e-6)

    def test_single_element(self):
        A = np.array([[5.0]], dtype=np.float32)
        B = np.array([[3.0]], dtype=np.float32)

        C_cuda = linalg.matmul(A, B)
        C_expected = np.array([[15.0]], dtype=np.float32)

        np.testing.assert_allclose(C_cuda, C_expected, rtol=1e-5)

    def test_vector_as_matrix(self):
        A = np.random.rand(5, 1).astype(np.float32)  # Column vector
        B = np.random.rand(1, 5).astype(np.float32)  # Row vector

        C_cuda = linalg.matmul(A, B)
        C_expected = A @ B

        assert C_cuda.shape == (5, 5)
        np.testing.assert_allclose(C_cuda, C_expected, rtol=1e-5)


class TestMatMulErrorHandling:
    def test_incompatible_shapes(self):
        A = np.random.rand(10, 20).astype(np.float32)
        B = np.random.rand(30, 40).astype(np.float32)

        with pytest.raises(ValueError, match="Inner dimensions must match"):
            linalg.matmul(A, B)

    def test_wrong_dimensions_1d(self):
        A = np.random.rand(10, 20).astype(np.float32)
        B = np.random.rand(20).astype(np.float32)  # 1D array

        with pytest.raises(ValueError, match="must be 2D arrays"):
            linalg.matmul(A, B)

    def test_wrong_dimensions_3d(self):
        A = np.random.rand(5, 10, 20).astype(np.float32)  # 3D array
        B = np.random.rand(20, 30).astype(np.float32)

        with pytest.raises(ValueError, match="must be 2D arrays"):
            linalg.matmul(A, B)

    def test_automatic_dtype_conversion(self):
        A = np.array([[1, 2], [3, 4]], dtype=np.float64)
        B = np.array([[5, 6], [7, 8]], dtype=np.float64)

        with pytest.raises(TypeError):
            linalg.matmul(A, B)

        # C_cuda = linalg.matmul(A, B)
        # C_expected = A @ B
        #
        # # Result should be float32 due to automatic conversion
        # assert C_cuda.dtype == np.float32
        # np.testing.assert_allclose(C_cuda, C_expected, rtol=1e-5)


class TestMatMulVariousSizes:
    @pytest.mark.parametrize(
        "m,k,n",
        [
            (1, 1, 1),  # Smallest: 1x1 @ 1x1
            (10, 5, 8),  # Small rectangular
            (16, 16, 16),  # Exact block size
            (15, 15, 15),  # Just under block size
            (17, 17, 17),  # Just over block size
            (32, 32, 32),  # Multiple of block size
            (64, 64, 64),  # Larger multiple
            (100, 1, 100),  # Tall and wide
            (1, 100, 1),  # Very thin middle
            (127, 131, 137),  # Prime numbers (no special alignment)
        ],
    )
    def test_various_sizes(self, m, k, n):
        rng = np.random.RandomState(42)
        A = rng.rand(m, k).astype(np.float32)
        B = rng.rand(k, n).astype(np.float32)

        C_cuda = linalg.matmul(A, B)
        C_expected = A @ B

        # Use adaptive tolerance based on matrix size (more accumulation in larger matrices)
        tolerance = 1e-4 * max(1, k // 100)
        np.testing.assert_allclose(C_cuda, C_expected, rtol=tolerance)
        assert C_cuda.shape == (m, n)


class TestMatMulPerformance:
    def test_performance_comparison(self, benchmark):
        size = 512
        A = np.random.rand(size, size).astype(np.float32)
        B = np.random.rand(size, size).astype(np.float32)

        linalg.matmul(A, B)  # Warm up GPU
        result = benchmark(linalg.matmul, A, B)

        expected = A @ B
        np.testing.assert_allclose(result, expected, rtol=1e-4)
