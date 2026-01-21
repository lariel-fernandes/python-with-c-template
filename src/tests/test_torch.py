import pytest
import torch
from my_proj.torch import MatmulLayer, matmul


class TestTorchMatmulCPU:
    """Test custom matmul on CPU."""

    def test_basic_matmul(self):
        """Test basic matrix multiplication on CPU."""
        A = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        B = torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32)

        C = matmul(A, B)
        C_expected = torch.matmul(A, B)

        assert C.shape == (2, 2)
        assert C.dtype == torch.float32
        assert torch.allclose(C, C_expected, rtol=1e-5)

    def test_rectangular_matrices(self):
        """Test with rectangular matrices."""
        A = torch.randn(10, 20, dtype=torch.float32)
        B = torch.randn(20, 30, dtype=torch.float32)

        C = matmul(A, B)
        C_expected = torch.matmul(A, B)

        assert C.shape == (10, 30)
        assert torch.allclose(C, C_expected, rtol=1e-4)

    def test_large_matrices(self):
        """Test with larger matrices."""
        A = torch.randn(100, 50, dtype=torch.float32)
        B = torch.randn(50, 80, dtype=torch.float32)

        C = matmul(A, B)
        C_expected = torch.matmul(A, B)

        assert C.shape == (100, 80)
        assert torch.allclose(C, C_expected, rtol=1e-2)

    def test_identity_matrix(self):
        """Test multiplication with identity matrix."""
        A = torch.randn(50, 50, dtype=torch.float32)
        I = torch.eye(50, dtype=torch.float32)

        C = matmul(A, I)

        assert torch.allclose(C, A, rtol=1e-5)


class TestTorchMatmulCUDA:
    """Test custom matmul on CUDA."""

    @pytest.fixture(autouse=True)
    def check_cuda(self):
        """Skip all tests in this class if CUDA is not available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

    def test_basic_matmul_cuda(self):
        """Test basic matrix multiplication on CUDA."""
        A = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32, device="cuda")
        B = torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32, device="cuda")

        C = matmul(A, B)
        C_expected = torch.matmul(A, B)

        assert C.device.type == "cuda"
        assert C.shape == (2, 2)
        assert C.dtype == torch.float32
        assert torch.allclose(C, C_expected, rtol=1e-5)

    def test_rectangular_matrices_cuda(self):
        """Test with rectangular matrices on CUDA."""
        A = torch.randn(10, 20, dtype=torch.float32, device="cuda")
        B = torch.randn(20, 30, dtype=torch.float32, device="cuda")

        C = matmul(A, B)
        C_expected = torch.matmul(A, B)

        assert C.device.type == "cuda"
        assert C.shape == (10, 30)
        assert torch.allclose(C, C_expected, rtol=1e-3)

    def test_large_matrices_cuda(self):
        """Test with larger matrices on CUDA."""
        A = torch.randn(256, 128, dtype=torch.float32, device="cuda")
        B = torch.randn(128, 512, dtype=torch.float32, device="cuda")

        C = matmul(A, B)
        C_expected = torch.matmul(A, B)

        assert C.device.type == "cuda"
        assert C.shape == (256, 512)
        assert torch.allclose(C, C_expected, rtol=1e-4)

    def test_various_sizes_cuda(self):
        """Test various matrix sizes on CUDA."""
        sizes = [(16, 16, 16), (15, 15, 15), (17, 17, 17), (100, 1, 100)]

        for m, k, n in sizes:
            A = torch.randn(m, k, dtype=torch.float32, device="cuda")
            B = torch.randn(k, n, dtype=torch.float32, device="cuda")

            C = matmul(A, B)
            C_expected = torch.matmul(A, B)

            assert C.shape == (m, n)
            assert torch.allclose(C, C_expected, rtol=1e-3)


class TestTorchMatmulAutograd:
    """Test autograd functionality."""

    def test_backward_cpu(self):
        """Test backward pass on CPU."""
        A = torch.randn(10, 20, dtype=torch.float32, requires_grad=True)
        B = torch.randn(20, 30, dtype=torch.float32, requires_grad=True)

        C = matmul(A, B)
        loss = C.sum()
        loss.backward()

        assert A.grad is not None
        assert B.grad is not None
        assert A.grad.shape == A.shape
        assert B.grad.shape == B.shape

    def test_gradcheck_cpu(self):
        """Test gradients using torch.autograd.gradcheck on CPU."""
        A = torch.randn(5, 10, dtype=torch.float64, requires_grad=True)
        B = torch.randn(10, 8, dtype=torch.float64, requires_grad=True)

        assert torch.autograd.gradcheck(
            lambda *args: torch.matmul(*args).double(),
            inputs=(A, B),
            eps=1e-4,
            atol=1e-3,
        )

    def test_gradient_correctness_cpu(self):
        """Verify gradient correctness by comparing with PyTorch's matmul."""
        A = torch.randn(10, 20, dtype=torch.float32, requires_grad=True)
        B = torch.randn(20, 30, dtype=torch.float32, requires_grad=True)

        # Compute with custom matmul
        C_custom = matmul(A, B)
        loss_custom = C_custom.sum()
        loss_custom.backward()
        grad_A_custom = A.grad.clone()
        grad_B_custom = B.grad.clone()

        # Reset gradients
        A.grad.zero_()
        B.grad.zero_()

        # Compute with PyTorch matmul
        C_torch = torch.matmul(A, B)
        loss_torch = C_torch.sum()
        loss_torch.backward()
        grad_A_torch = A.grad
        grad_B_torch = B.grad

        # Compare gradients
        assert torch.allclose(grad_A_custom, grad_A_torch, rtol=1e-4)
        assert torch.allclose(grad_B_custom, grad_B_torch, rtol=1e-4)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_backward_cuda(self):
        """Test backward pass on CUDA."""
        A = torch.randn(10, 20, dtype=torch.float32, device="cuda", requires_grad=True)
        B = torch.randn(20, 30, dtype=torch.float32, device="cuda", requires_grad=True)

        C = matmul(A, B)
        loss = C.sum()
        loss.backward()

        assert A.grad is not None
        assert B.grad is not None
        assert A.grad.device.type == "cuda"
        assert B.grad.device.type == "cuda"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gradient_correctness_cuda(self):
        """Verify gradient correctness on CUDA."""
        A = torch.randn(10, 20, dtype=torch.float32, device="cuda", requires_grad=True)
        B = torch.randn(20, 30, dtype=torch.float32, device="cuda", requires_grad=True)

        # Compute with custom matmul
        C_custom = matmul(A, B)
        loss_custom = C_custom.sum()
        loss_custom.backward()
        grad_A_custom = A.grad.clone()
        grad_B_custom = B.grad.clone()

        # Reset gradients
        A.grad.zero_()
        B.grad.zero_()

        # Compute with PyTorch matmul
        C_torch = torch.matmul(A, B)
        loss_torch = C_torch.sum()
        loss_torch.backward()
        grad_A_torch = A.grad
        grad_B_torch = B.grad

        # Compare gradients
        assert torch.allclose(grad_A_custom, grad_A_torch, rtol=1e-4)
        assert torch.allclose(grad_B_custom, grad_B_torch, rtol=1e-4)


class TestMatmulLayer:
    """Test MatmulLayer nn.Module wrapper."""

    def test_layer_basic(self):
        """Test basic layer usage."""
        layer = MatmulLayer()
        A = torch.randn(10, 20, dtype=torch.float32)
        B = torch.randn(20, 30, dtype=torch.float32)

        C = layer(A, B)
        C_expected = torch.matmul(A, B)

        assert torch.allclose(C, C_expected, rtol=1e-3)

    def test_layer_in_model(self):
        """Test layer as part of a model."""

        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.matmul = MatmulLayer()
                self.linear = torch.nn.Linear(30, 10)

            def forward(self, A, B):
                x = self.matmul(A, B)
                return self.linear(x)

        model = SimpleModel()
        A = torch.randn(5, 20, dtype=torch.float32)
        B = torch.randn(20, 30, dtype=torch.float32)

        output = model(A, B)
        assert output.shape == (5, 10)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_layer_device_transfer(self):
        """Test that layer works after moving to different devices."""
        layer = MatmulLayer()

        # Test on CPU
        A_cpu = torch.randn(10, 20, dtype=torch.float32)
        B_cpu = torch.randn(20, 30, dtype=torch.float32)
        C_cpu = layer(A_cpu, B_cpu)
        assert C_cpu.device.type == "cpu"

        # Test on CUDA
        A_cuda = torch.randn(10, 20, dtype=torch.float32, device="cuda")
        B_cuda = torch.randn(20, 30, dtype=torch.float32, device="cuda")
        C_cuda = layer(A_cuda, B_cuda)
        assert C_cuda.device.type == "cuda"

    def test_layer_training_loop(self):
        """Test layer in a simple training loop."""

        class TrainableModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.matmul = MatmulLayer()
                self.weight = torch.nn.Parameter(torch.randn(30, 10))

            def forward(self, A, B):
                x = self.matmul(A, B)
                return torch.matmul(x, self.weight)

        model = TrainableModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # Simple training loop
        for _ in range(5):
            A = torch.randn(5, 20, dtype=torch.float32)
            B = torch.randn(20, 30, dtype=torch.float32)
            target = torch.randn(5, 10, dtype=torch.float32)

            optimizer.zero_grad()
            output = model(A, B)
            loss = torch.nn.functional.mse_loss(output, target)
            loss.backward()
            optimizer.step()

        assert True  # If we got here, training loop works


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_incompatible_shapes(self):
        """Test that incompatible shapes raise an error."""
        A = torch.randn(10, 20, dtype=torch.float32)
        B = torch.randn(30, 40, dtype=torch.float32)

        with pytest.raises(RuntimeError, match="Inner dimensions must match"):
            matmul(A, B)

    def test_wrong_dimensions(self):
        """Test that wrong number of dimensions raises an error."""
        A = torch.randn(10, 20, dtype=torch.float32)
        B = torch.randn(20, dtype=torch.float32)  # 1D tensor

        with pytest.raises(RuntimeError, match="must be a 2D tensor"):
            matmul(A, B)

    def test_device_mismatch(self):
        """Test that device mismatch raises an error."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        A = torch.randn(10, 20, dtype=torch.float32, device="cpu")
        B = torch.randn(20, 30, dtype=torch.float32, device="cuda")

        with pytest.raises(RuntimeError, match="same device"):
            matmul(A, B)

    def test_non_contiguous(self):
        """Test that non-contiguous tensors raise an error."""
        A = torch.randn(10, 20, dtype=torch.float32).t()  # Non-contiguous
        B = torch.randn(10, 30, dtype=torch.float32)

        with pytest.raises(RuntimeError, match="must be contiguous"):
            matmul(A, B)
