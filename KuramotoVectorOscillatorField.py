import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import fetch_openml  # type: ignore
from scipy.ndimage import zoom  # type: ignore

EPSILON: float = 1e-9


class KuramotoVectorOscillatorField:
    def __init__(
        self,
        h: int,
        w: int,
        dims: int = 4,
        d_t: float = 0.01,
        k_coupling: float = 0.5,
        k_omega: float = 1.0,
        k_bias: float = 0.3,
        spike_thresh: float = 0.01,
        initial_phases: NDArray | None = None,
    ) -> None:
        self.H, self.W, self.D = h, w, dims
        self.dt = d_t
        self.k_coupling = k_coupling
        self.k_omega = k_omega
        self.k_bias = k_bias
        self.spike_thresh = spike_thresh

        if initial_phases is None:
            initial_phases = (
                np.random.rand(H, W, D) * 2 * np.pi
            )  # Oscillator state: unit complex vectors (H, W, D)

        # Oscillator state: unit complex vectors (H, W, D)
        phase = initial_phases
        self.z = np.exp(1j * phase)  # Complex array
        self.z_prev = self.z.copy()

        # External bias (input)
        self.c = self.z.copy()

        # Skew-symmetric omega for cross-dimensional coupling (D, D)
        omega = np.random.randn(D, D)
        self.omega = omega - omega.T  # Skew-symmetric

    def neighbor_sum(self, z: NDArray) -> NDArray:
        # 6-neighbor coupling with zero-padding
        z_padded = np.pad(
            z, ((1, 1), (1, 1), (1, 1)), mode="constant", constant_values=0
        )
        rolled = (
            z_padded[1:-1, 1:-1, :-2]
            + z_padded[1:-1, 1:-1, 2:]  # D
            + z_padded[1:-1, :-2, 1:-1]
            + z_padded[1:-1, 2:, 1:-1]  # W
            + z_padded[:-2, 1:-1, 1:-1]
            + z_padded[2:, 1:-1, 1:-1]  # H
        )
        return rolled

    def normalize(self, z: NDArray) -> NDArray:
        return z / (np.abs(z) + EPSILON)

    def step(self, evolve_c: bool = False) -> NDArray:
        self.z_prev = self.z.copy()

        # 1. Local spatial coupling
        z_neighbors = self.neighbor_sum(self.z)

        # 2. Cross-dimensional omega interaction
        # z: (H, W, D), omega: (D, D) -> einsum 'ij,hwj->hwi'
        z_omega = np.einsum("ij,hwj->hwi", self.omega, self.z)

        # 3. Input bias direction
        bias = self.c - self.z

        # 4. Total update
        dz = (
            self.k_coupling * (z_neighbors - self.z)
            + self.k_omega * z_omega
            + self.k_bias * bias
        )

        # 5. Euler update + normalize
        self.z = self.normalize(self.z + self.dt * dz)

        # 6. Compute spikes (phase velocity)
        d_theta = np.angle(self.z * np.conj(self.z_prev))
        spike_activity = (np.abs(d_theta) > self.spike_thresh).astype(np.float32)

        # 7. Evolve external input field
        if evolve_c:
            dc = 0.1 * (self.z - self.c)
            self.c = self.normalize(self.c + self.dt * dc)

        return spike_activity

    def run(
        self, steps: int = 100, evolve_c: bool = False, return_history: bool = False
    ) -> tuple[NDArray, NDArray] | tuple[list[NDArray], list[NDArray]]:
        z_history_list = []
        spike_history_list = []
        spikes_generated = np.zeros((self.H, self.W, self.D))

        for _ in range(steps):
            spikes_generated = self.step(evolve_c=evolve_c)
            if return_history:
                z_history_list.append(self.z.copy())
                spike_history_list.append(spikes_generated.copy())
        return (
            (z_history_list, spike_history_list)
            if return_history
            else (self.z, spikes_generated)
        )


# Load MNIST digit
DIGIT = "6"
X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
the_digit = X[y == DIGIT][0].reshape(28, 28)
the_digit = (the_digit / 255.0) * 2 * np.pi  # Scale 0-1 to 0-2π

# Invert the polarity of the image
the_digit = 2 * np.pi - the_digit

# Pad the input image with a border of zeros
PAD_SIZE = 2
digit_3_padded = np.pad(
    the_digit, pad_width=PAD_SIZE, mode="constant", constant_values=0
)  # 32x32

# Resize to 16x16
H, W, D = 16 + 2 * PAD_SIZE, 16 + 2 * PAD_SIZE, 16  # 20x20x16 with padding
digit_3_resized = zoom(digit_3_padded, (H - 2 * PAD_SIZE) / 32, order=1)  # 16x16

# Create perturbation (20x20x1 slice, repeat across D)
perturbation = np.zeros((H, W, D), dtype=np.float32)
perturbation[PAD_SIZE:-PAD_SIZE, PAD_SIZE:-PAD_SIZE, 0] = (
    digit_3_resized  # Center the '3'
)
for d in range(1, D):
    perturbation[:, :, d] = perturbation[:, :, 0]  # Repeat across depth

# Initialize field
field = KuramotoVectorOscillatorField(
    h=H,
    w=W,
    dims=D,
    d_t=0.01,
    k_coupling=0.12,
    k_omega=0.12,
    k_bias=1.25,
    spike_thresh=0.00035,
)

# Set perturbation
field.c = np.exp(1j * perturbation)

# Run for 2048 steps
z_history, spike_history = field.run(steps=2048, evolve_c=True, return_history=True)

# Visualize
for t in [
    0,
    1,
    10,
    50,
    100,
    125,
    150,
    200,
    250,
    275,
    300,
    325,
    350,
    375,
    400,
    450,
    500,
    750,
    1000,
    1250,
    1500,
    1750,
    2000,
]:
    theta = np.angle(z_history[t]).mean(axis=2)  # 20x20
    spikes = spike_history[t].mean(axis=2)  # 20x20
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.imshow(digit_3_padded, cmap="gray")
    plt.title(f"Input '{DIGIT}' (Padded)")
    plt.subplot(132)
    plt.imshow(theta, cmap="hsv")
    plt.title(f"Theta (t={t})")
    plt.subplot(133)
    plt.imshow(spikes, cmap="binary")
    plt.title(f"Spikes (t={t})")
    plt.show()
    plt.tight_layout()
