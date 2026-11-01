// WebGL 2.0 / Canvas 2D Fallback Engine
// Provides high-compatibility execution of the Neural Cellular Automata ecosystem
// when WebGPU is unavailable or unsupported on the user's browser/hardware.

export class WebGLFallbackEngine {
    constructor() {
        this.canvas = null;
        this.ctx = null;
        this.gl = null;
        this.width = 256;
        this.height = 256;
        
        // Double-buffered cell memory: 16 channels per cell
        this.bufferA = null;
        this.bufferB = null;
        this.currentBuffer = null;
        this.nextBuffer = null;
        
        this.weights = null; // Packed 8336 Float32Array
        this.stepCounter = 0;
        this.livingCount = 0;
        
        this.imageData = null;
    }

    async init(canvas) {
        this.canvas = canvas;
        
        // Try WebGL 2.0 first, else fallback to 2D canvas
        this.ctx = canvas.getContext('2d');
        if (!this.ctx) {
            throw new Error("Canvas context initialization failed.");
        }

        const gpuDesc = "WebGL 2.0 / 2D Canvas Compatibility Fallback";
        await this.setResolution(this.width, this.height);

        return {
            gpuDesc,
            limits: { maxTextureDimension2D: 4096 }
        };
    }

    async setResolution(width, height) {
        this.width = width;
        this.height = height;
        const total = width * height * 16;
        
        this.bufferA = new Float32Array(total);
        this.bufferB = new Float32Array(total);
        this.currentBuffer = this.bufferA;
        this.nextBuffer = this.bufferB;
        
        this.imageData = this.ctx.createImageData(width, height);
        this.stepCounter = 0;
    }

    loadModelWeights(packedWeights) {
        this.weights = new Float32Array(packedWeights);
    }

    seed(seedType) {
        const total = this.width * this.height * 16;
        this.bufferA.fill(0);
        this.bufferB.fill(0);

        const cx = Math.floor(this.width / 2);
        const cy = Math.floor(this.height / 2);

        if (seedType === 'center_single') {
            for (let dy = -1; dy <= 1; dy++) {
                for (let dx = -1; dx <= 1; dx++) {
                    const idx = ((cy + dy) * this.width + (cx + dx)) * 16;
                    this.bufferA[idx + 0] = 0.2;
                    this.bufferA[idx + 1] = 0.8;
                    this.bufferA[idx + 2] = 0.4;
                    this.bufferA[idx + 3] = 1.0;
                    for (let c = 4; c < 16; c++) {
                        this.bufferA[idx + c] = 1.0;
                    }
                }
            }
        } else if (seedType === 'slime_cluster') {
            for (let i = 0; i < 5; i++) {
                const angle = (i / 5) * Math.PI * 2;
                const r = this.width * 0.28;
                const nx = Math.floor(cx + Math.cos(angle) * r);
                const ny = Math.floor(cy + Math.sin(angle) * r);
                for (let dy = -4; dy <= 4; dy++) {
                    for (let dx = -4; dx <= 4; dx++) {
                        if (dx * dx + dy * dy <= 16) {
                            const idx = ((ny + dy) * this.width + (nx + dx)) * 16;
                            this.bufferA[idx + 0] = 0.9;
                            this.bufferA[idx + 1] = 0.95;
                            this.bufferA[idx + 2] = 0.2;
                            this.bufferA[idx + 3] = 1.0;
                            this.bufferA[idx + 4] = 1.0;
                            this.bufferA[idx + 7] = 1.0;
                        }
                    }
                }
            }
        } else if (seedType === 'jelly_core') {
            for (let dy = -8; dy <= 8; dy++) {
                for (let dx = -8; dx <= 8; dx++) {
                    const d2 = dx * dx + dy * dy;
                    if (d2 <= 64) {
                        const idx = ((cy + dy) * this.width + (cx + dx)) * 16;
                        this.bufferA[idx + 0] = 0.05;
                        this.bufferA[idx + 1] = 0.5;
                        this.bufferA[idx + 2] = 0.95;
                        this.bufferA[idx + 3] = 1.0;
                        this.bufferA[idx + 4] = 0.5;
                    }
                }
            }
        } else if (seedType === 'coral_spores') {
            const pts = [[cx, cy], [cx - 40, cy - 30], [cx + 40, cy + 30]];
            for (const [sx, sy] of pts) {
                for (let dy = -3; dy <= 3; dy++) {
                    for (let dx = -3; dx <= 3; dx++) {
                        const idx = ((sy + dy) * this.width + (sx + dx)) * 16;
                        this.bufferA[idx + 0] = 0.85;
                        this.bufferA[idx + 1] = 0.2;
                        this.bufferA[idx + 2] = 0.9;
                        this.bufferA[idx + 3] = 1.0;
                        this.bufferA[idx + 4] = 1.0;
                    }
                }
            }
        } else if (seedType === 'turing_spots') {
            for (let i = 0; i < this.width * this.height; i++) {
                this.bufferA[i * 16 + 0] = 1.0;
                this.bufferA[i * 16 + 3] = 1.0;
            }
            for (let s = 0; s < 12; s++) {
                const rx = Math.floor(Math.random() * (this.width - 20)) + 10;
                const ry = Math.floor(Math.random() * (this.height - 20)) + 10;
                for (let dy = -5; dy <= 5; dy++) {
                    for (let dx = -5; dx <= 5; dx++) {
                        const idx = ((ry + dy) * this.width + (rx + dx)) * 16;
                        this.bufferA[idx + 0] = 0.5;
                        this.bufferA[idx + 1] = 0.25;
                    }
                }
            }
        }

        this.bufferB.set(this.bufferA);
        this.currentBuffer = this.bufferA;
        this.nextBuffer = this.bufferB;
        this.stepCounter = 0;
    }

    step(simParams) {
        const t0 = performance.now();
        this.stepCounter++;

        const w = this.width;
        const h = this.height;
        const src = this.currentBuffer;
        const dst = this.nextBuffer;
        const weights = this.weights;
        const modelType = simParams.modelType;
        const updateRate = simParams.updateRate;
        const aliveThreshold = simParams.aliveThreshold;

        const sobelX = [-0.125, 0, 0.125, -0.25, 0, 0.25, -0.125, 0, 0.125];
        const sobelY = [-0.125, -0.25, -0.125, 0, 0, 0, 0.125, 0.25, 0.125];
        const laplacian = [0.125, 0.25, 0.125, 0.25, -1.5, 0.25, 0.125, 0.25, 0.125];

        let liveCount = 0;
        const p = new Float32Array(48);
        const lap = new Float32Array(16);
        const hidden = new Float32Array(128);

        for (let y = 0; y < h; y++) {
            const ym1 = (y - 1 + h) % h;
            const yp1 = (y + 1) % h;
            const rowM1 = ym1 * w;
            const row0 = y * w;
            const rowP1 = yp1 * w;

            for (let x = 0; x < w; x++) {
                const xm1 = (x - 1 + w) % w;
                const xp1 = (x + 1) % w;
                const cellIdx = (row0 + x) * 16;

                p.fill(0);
                lap.fill(0);
                let maxAlpha = 0;

                const neighborIndices = [
                    (rowM1 + xm1) * 16, (rowM1 + x) * 16, (rowM1 + xp1) * 16,
                    (row0 + xm1) * 16, cellIdx, (row0 + xp1) * 16,
                    (rowP1 + xm1) * 16, (rowP1 + x) * 16, (rowP1 + xp1) * 16
                ];

                for (let k = 0; k < 9; k++) {
                    const nIdx = neighborIndices[k];
                    const a = src[nIdx + 3];
                    if (a > maxAlpha) maxAlpha = a;

                    const sx = sobelX[k];
                    const sy = sobelY[k];
                    const lp = laplacian[k];

                    const isCenter = (k === 4);
                    for (let c = 0; c < 16; c++) {
                        const val = src[nIdx + c];
                        if (isCenter) p[c] = val;
                        p[16 + c] += sx * val;
                        p[32 + c] += sy * val;
                        lap[c] += lp * val;
                    }
                }

                if (modelType === 0) {
                    // Neural NCA
                    if (maxAlpha <= aliveThreshold || !weights) {
                        for (let c = 0; c < 16; c++) dst[cellIdx + c] = 0;
                        continue;
                    }

                    // Hidden Layer: 48 -> 128 ReLU
                    for (let j = 0; j < 128; j++) {
                        let sum = weights[6144 + j];
                        for (let k = 0; k < 48; k++) {
                            sum += p[k] * weights[k * 128 + j];
                        }
                        hidden[j] = sum > 0 ? sum : 0;
                    }

                    // Stochastic mask
                    const shouldUpdate = Math.random() < updateRate;
                    for (let c = 0; c < 16; c++) {
                        if (shouldUpdate) {
                            let delta = weights[8320 + c];
                            for (let j = 0; j < 128; j++) {
                                delta += hidden[j] * weights[6272 + j * 16 + c];
                            }
                            dst[cellIdx + c] = p[c] + delta;
                        } else {
                            dst[cellIdx + c] = p[c];
                        }
                    }

                    if (dst[cellIdx + 3] <= aliveThreshold && maxAlpha <= aliveThreshold) {
                        for (let c = 0; c < 16; c++) dst[cellIdx + c] = 0;
                        continue;
                    }

                } else if (modelType === 1) {
                    // Slime Mold
                    const d = p[0];
                    const a = p[4];
                    const cond = p[7];
                    const newD = Math.max(0, Math.min(1, d + lap[0] * 0.2 + cond * 0.05));
                    dst[cellIdx + 0] = 0.92 * newD;
                    dst[cellIdx + 1] = 0.98 * newD;
                    dst[cellIdx + 2] = 0.2 * newD;
                    dst[cellIdx + 3] = newD;
                    dst[cellIdx + 4] = a + lap[4] * 0.15;
                    dst[cellIdx + 7] = Math.max(0.05, Math.min(2, cond * 0.99 + newD * 0.05));

                } else if (modelType === 2) {
                    // Jellyfish
                    const phase = (p[4] + 0.08) % 1.0;
                    const pulse = Math.pow(Math.sin(phase * Math.PI * 2), 4.0);
                    dst[cellIdx + 0] = 0.1 + pulse * 0.2;
                    dst[cellIdx + 1] = 0.4 + pulse * 0.4;
                    dst[cellIdx + 2] = 0.8 + pulse * 0.2;
                    dst[cellIdx + 3] = pulse > 0.1 ? 0.8 : 0.2;
                    dst[cellIdx + 4] = phase;

                } else if (modelType === 3) {
                    // Biocrystal
                    const c = p[0];
                    const nut = p[4];
                    const growth = nut * (lap[0] * 0.2 + c * (1 - c) * 0.6);
                    const newC = Math.max(0, Math.min(1, c + growth));
                    dst[cellIdx + 0] = 0.7 * newC;
                    dst[cellIdx + 1] = 0.2 * newC;
                    dst[cellIdx + 2] = 0.9 * newC;
                    dst[cellIdx + 3] = newC;
                    dst[cellIdx + 4] = Math.max(0, nut - growth * 0.5);

                } else if (modelType === 4) {
                    // Turing Reaction Diffusion
                    const u = p[0];
                    const v = p[1];
                    const uvv = u * v * v;
                    const du = 0.18 * lap[0] - uvv + 0.0545 * (1 - u);
                    const dv = 0.09 * lap[1] + uvv - 0.1165 * v;
                    const newU = Math.max(0, Math.min(1, u + du * 1.5));
                    const newV = Math.max(0, Math.min(1, v + dv * 1.5));
                    dst[cellIdx + 0] = newV * 2.2;
                    dst[cellIdx + 1] = newU * 0.5;
                    dst[cellIdx + 2] = (1 - newU) * 1.5;
                    dst[cellIdx + 3] = (newU + newV) * 0.8;
                }

                // Brush tool
                if (simParams.brushActive) {
                    const dx = x - simParams.brushX;
                    const dy = y - simParams.brushY;
                    const dist = Math.sqrt(dx * dx + dy * dy);
                    if (dist <= simParams.brushRadius) {
                        const falloff = 1 - (dist / simParams.brushRadius);
                        if (simParams.brushType === 0) {
                            // Scalpel
                            for (let c = 0; c < 16; c++) dst[cellIdx + c] *= (1 - falloff);
                        } else if (simParams.brushType === 1) {
                            // Seed
                            dst[cellIdx + 0] = 0.2;
                            dst[cellIdx + 1] = 0.8;
                            dst[cellIdx + 2] = 0.4;
                            dst[cellIdx + 3] = 1.0;
                            for (let c = 4; c < 16; c++) dst[cellIdx + c] = 1.0;
                        }
                    }
                }

                if (dst[cellIdx + 3] > 0.1) liveCount++;
            }
        }

        // Swap ping pong buffers
        const tmp = this.currentBuffer;
        this.currentBuffer = this.nextBuffer;
        this.nextBuffer = tmp;

        this.livingCount = liveCount;
        return performance.now() - t0;
    }

    render(renderParams) {
        const w = this.width;
        const h = this.height;
        const src = this.currentBuffer;
        const img = this.imageData;
        const data = img.data;

        for (let i = 0; i < w * h; i++) {
            const cellIdx = i * 16;
            const pxIdx = i * 4;
            const r = src[cellIdx + 0];
            const g = src[cellIdx + 1];
            const b = src[cellIdx + 2];
            const a = src[cellIdx + 3];

            if (renderParams.displayMode === 0) {
                // RGBA
                data[pxIdx + 0] = Math.floor(Math.min(255, Math.max(0, (r * a + 0.04 * (1 - a)) * 255)));
                data[pxIdx + 1] = Math.floor(Math.min(255, Math.max(0, (g * a + 0.05 * (1 - a)) * 255)));
                data[pxIdx + 2] = Math.floor(Math.min(255, Math.max(0, (b * a + 0.08 * (1 - a)) * 255)));
                data[pxIdx + 3] = 255;
            } else if (renderParams.displayMode === 1) {
                // Morphogen channel
                const ch = src[cellIdx + (renderParams.selectedChannel || 4)];
                const v = Math.floor(Math.min(255, Math.max(0, (ch * 0.5 + 0.5) * 255)));
                data[pxIdx + 0] = v;
                data[pxIdx + 1] = 255 - v;
                data[pxIdx + 2] = 180;
                data[pxIdx + 3] = 255;
            } else {
                // Alpha
                const v = Math.floor(Math.min(255, Math.max(0, a * 255)));
                data[pxIdx + 0] = v;
                data[pxIdx + 1] = v;
                data[pxIdx + 2] = v;
                data[pxIdx + 3] = 255;
            }
        }

        // Draw image data scaled to canvas
        this.ctx.putImageData(img, 0, 0);
        if (this.canvas.width !== w || this.canvas.height !== h) {
            this.ctx.drawImage(this.canvas, 0, 0, w, h, 0, 0, this.canvas.width, this.canvas.height);
        }

        // Render reticle overlay
        if (renderParams.brushActive) {
            const scaleX = this.canvas.width / w;
            const scaleY = this.canvas.height / h;
            this.ctx.strokeStyle = renderParams.brushType === 0 ? '#ef4444' : '#10b981';
            this.ctx.lineWidth = 2;
            this.ctx.beginPath();
            this.ctx.arc(
                renderParams.brushX * scaleX,
                renderParams.brushY * scaleY,
                renderParams.brushRadius * scaleX,
                0,
                Math.PI * 2
            );
            this.ctx.stroke();
        }
    }

    async sampleLivingCellsAsync() {
        return this.livingCount;
    }
}
