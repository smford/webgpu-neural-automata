# WebGPU Neural Cellular Automata

[![Deploy GitHub Pages](https://github.com/smford/webgpu-neural-automata/actions/workflows/deploy-pages.yml/badge.svg)](https://github.com/smford/webgpu-neural-automata/actions/workflows/deploy-pages.yml)
[![Live Demo](https://img.shields.io/badge/Live_Site-stephenford.org%2Fwebgpu--neural--automata-06b6d4?style=flat&logo=webgpu)](https://stephenford.org/webgpu-neural-automata/)

> **Live Application:** [https://stephenford.org/webgpu-neural-automata/](https://stephenford.org/webgpu-neural-automata/)

An interactive, high-performance **Neural Cellular Automata (NCA)** simulator executing self-repairing synthetic biological morphogenesis directly on the GPU using **WebGPU compute pipelines** and **WGSL** (WebGPU Shading Language).

---

## Concept

Neural Cellular Automata merge classical discrete dynamical cellular automata (like Conway's Game of Life) with deep artificial neural networks. In this simulator, each cell on a selectable $256 \times 256$ to $2048 \times 2048$ grid maintains a state vector of **16 continuous floating-point channels**:
- **Channels 0–2 (RGB):** Visual phenotypic display.
- **Channel 3 ($\alpha$):** Living cell mask & physical biomass opacity.
- **Channels 4–15:** 12 hidden chemical morphogen channels that store positional coordinates, differentiation states, and signaling cascades.

Unlike classical hard-coded cellular rules, every single cell independently runs an identical **2-layer Convolutional Multilayer Perceptron (MLP)** that senses its local chemical gradient through 3×3 spatial convolutions and calculates its next metabolic state update purely in parallel on the GPU.

When damaged—such as severing limbs or carving out wounds with the interactive **Damage Scalpel**—the surrounding living cells detect the disrupted chemical gradients and coordinate dynamically to regrow, reconstruct lost patterns, and heal the organism in real time.

---

## Why It's Quirky

- **Emergent Biological Self-Repair Without Global Memory:** There is no master coordinator, no central template, and no CPU image buffers stored anywhere. The complete blueprint and repair instructions are decentralized and encoded entirely into the synaptic weights of the tiny neural network running in parallel inside every cell.
- **Zero-CPU Simulation Execution:** The entire forward pass—spatial convolutions, ReLU activations, dense matrix multiplications, stochastic asynchronous masking, and alive thresholding—executes inside a single unified WGSL compute shader pass at up to 120 FPS.
- **Seamless Universal Compatibility Fallback:** If WebGPU is not supported by the client browser or hardware, the engine automatically engages an optimized WebGL 2.0 / Canvas 2D fallback with a real-time status badge (`WebGPU Active` vs `WebGL 2.0 Active`).

---

## Architecture & Mathematical Foundation

```
 ┌────────────────────────────────────────────────────────────────────────┐
 │                      PING-PONG GPU STORAGE BUFFERS                     │
 │  Buffer A (f32 x 16 x W x H)  <======>  Buffer B (f32 x 16 x W x H)    │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │
                    ┌────────────────▼───────────────┐
                    │      PERCEPTION CONVOLUTION     │
                    │   3x3 Sobel-X, Sobel-Y, Ident   │
                    │      (48 Feature Channels)      │
                    └────────────────┬───────────────┘
                                     │
                    ┌────────────────▼───────────────┐
                    │     LAYER 1: DENSE + RELU      │
                    │   48 Inputs -> 128 Hidden      │
                    │      h = max(0, W1 * p + b1)   │
                    └────────────────┬───────────────┘
                                     │
                    ┌────────────────▼───────────────┐
                    │     LAYER 2: DENSE DELTA       │
                    │   128 Hidden -> 16 Outputs     │
                    │       delta = W2 * h + b2      │
                    └────────────────┬───────────────┘
                                     │
                    ┌────────────────▼───────────────┐
                    │    STOCHASTIC MASK & ALIVE     │
                    │   Hash(x, y, seed) < 50% rate  │
                    │   max(alpha_3x3) > 0.1 alive   │
                    └────────────────┬───────────────┘
                                     │
                    ┌────────────────▼───────────────┐
                    │    FULLSCREEN TEXTURE BLIT     │
                    │   Hardware Zoom, Pan & Reticle │
                    │       Canvas @ 120 FPS         │
                    └────────────────────────────────┘
```

### 1. Perception Step
Every cell samples its 8 neighbors in a $3 \times 3$ kernel to determine chemical concentration gradients across all 16 channels:
$$\mathbf{p}(x, y) = \begin{bmatrix} \mathbf{c}(x, y) \\ \mathbf{K}_x * \mathbf{c}(x, y) \\ \mathbf{K}_y * \mathbf{c}(x, y) \end{bmatrix} \in \mathbb{R}^{48}$$

Where the normalized Sobel gradient kernels $\mathbf{K}_x$ and $\mathbf{K}_y$ are:
$$\mathbf{K}_x = \frac{1}{8} \begin{bmatrix} -1 & 0 & +1 \\ -2 & 0 & +2 \\ -1 & 0 & +1 \end{bmatrix}, \quad \mathbf{K}_y = \frac{1}{8} \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ +1 & +2 & +1 \end{bmatrix}$$

### 2. Neural Update Step
The 48-dimensional perception vector passes through a two-layer dense perceptron:
$$\mathbf{h} = \text{ReLU}(\mathbf{W}_1 \mathbf{p} + \mathbf{b}_1), \quad \mathbf{h} \in \mathbb{R}^{128}$$
$$\mathbf{\Delta} = \mathbf{W}_2 \mathbf{h} + \mathbf{b}_2, \quad \mathbf{\Delta} \in \mathbb{R}^{16}$$

### 3. Stochastic Asynchronous Update Masking
Living biological cells do not fire in lockstep on a global clock. To simulate asynchronous growth and prevent synchronization artifacts, only a pseudo-random $\approx 50\%$ subset of cells update per compute dispatch using Hoskins' PRNG hash:
$$\mathbf{c}_{t+1} = \begin{cases} \mathbf{c}_t + \mathbf{\Delta} & \text{if } \text{hash}(x, y, \text{seed}) < \text{rate} \\ \mathbf{c}_t & \text{otherwise} \end{cases}$$

### 4. Alive Masking & Boundary Pruning
Spontaneous unseeded generation is prevented by enforcing living neighborhood constraints:
$$\alpha_{\max} = \max_{(dx, dy) \in \{-1, 0, 1\}^2} \mathbf{c}_{t}(x+dx, y+dy).\alpha$$
If $\alpha_{\max} \le 0.1$, the cell is considered dead space and its entire 16-channel state is zeroed.

---

## Curated Ecosystem Presets

1. **Self-Repairing Gecko:** The classic Distill.pub morphogenetic organism. When damaged with the scalpel, adjacent tissue senses morphological loss and rebuilds limbs, body walls, and skin patterns.
2. **Physarum Slime Mold:** A plasmodial foraging bio-network exhibiting protoplasmic streaming and vein optimization connecting nutrient nodes.
3. **Bioluminescent Jellyfish:** Rhythmic radial limit-cycle pacemaker contractions with mechanically stimulated luciferin luminescent flares and trailing tentacles.
4. **Alien Coral Biocrystal:** Anisotropic phase-field crystallization forming iridescent mineral coral facets with 6-fold dendritic symmetry.
5. **Turing Reaction-Diffusion:** Multi-scale Gray-Scott chemical reaction-diffusion waves generating labyrinthine patterns and self-replicating spots.
6. **Metamorphic Butterfly:** Bilateral wing morphogenesis demonstrating developmental symmetry.
7. **Immortal Planarian:** Pluripotent flatworm model organism capable of regenerating both head and tail from severed fragments.
8. **Tropical Reef Fish:** Dynamic pigment chromatophore patterning and fin regeneration.

---

## Interactive Studio Tools

- **Damage Scalpel / Eraser:** Click and drag across the organism to carve lacerations or amputate limbs and observe real-time neural regeneration.
- **Seed Pen:** Inoculate fresh embryonic stem cell clusters into empty substrate.
- **Chemical Mutator Brush:** Inject localized chemical noise into hidden morphogen channels ($c_4 \dots c_{15}$) to spark morphological mutations.
- **Nutrient Catalyst:** Fuel accelerated mitotic division and stimulate rapid healing.
- **Neural Synapse Mutation Lab:** Perturb the active model weight matrices $\mathbf{W}_1, \mathbf{W}_2$ in real time to observe evolutionary drift.
- **Hidden Morphogen Inspector:** View hidden channels 4 through 15 in false-color Turbo heatmaps.
- **Smooth Optical Zoom & Pan:** Hardware-accelerated camera navigation across the microscopic cellular colony.

---

## Local Development

To run the application locally without build tooling:

```bash
# Clone the repository
git clone https://github.com/smford/webgpu-neural-automata.git
cd webgpu-neural-automata

# Serve the static documentation directory
python3 -m http.server 8000 --directory docs
```

Open [http://localhost:8000](http://localhost:8000) in any browser supporting WebGPU (Google Chrome, Microsoft Edge, Safari 18+, or Firefox Nightly) or WebGL 2.0.

---

## Deployment & GitHub Actions

The repository is configured to deploy directly to GitHub Pages via `.github/workflows/deploy-pages.yml` on every push to the `main` branch. All site files reside under `./docs` with strictly relative paths to ensure zero 404s when hosted under subpaths.

---

## License

Apache-2.0. Neural model architectures adapted from the Distill.pub self-organizing systems project.
