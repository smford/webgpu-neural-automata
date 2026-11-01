// WebGPU Neural Cellular Automata Engine
// Hardware-accelerated compute pipeline using double-buffered storage buffers and WGSL shaders.

import { COMPUTE_SHADER, RENDER_SHADER } from './shaders.js';
import { unpackModelWeights } from './models.js';

export class WebGPUEngine {
    constructor() {
        this.device = null;
        this.adapter = null;
        this.context = null;
        this.canvas = null;
        this.format = null;
        
        this.width = 512;
        this.height = 512;
        
        this.bufferA = null;
        this.bufferB = null;
        this.currentBufferIdx = 0;
        
        this.simParamsBuffer = null;
        this.renderParamsBuffer = null;
        this.weightsBuffer = null;
        
        this.computePipeline = null;
        this.renderPipeline = null;
        
        this.computeBindGroups = [null, null];
        this.renderBindGroups = [null, null];
        
        this.stepCounter = 0;
        this.currentModelType = 0;
        this.lastLivingCount = 0;
        this.isStagingBusy = false;
        this.stagingBuffer = null;
    }

    async init(canvas) {
        if (!navigator.gpu) {
            throw new Error("WebGPU is not supported by your browser (navigator.gpu is undefined).");
        }

        this.canvas = canvas;
        this.adapter = await navigator.gpu.requestAdapter({
            powerPreference: 'high-performance'
        });

        if (!this.adapter) {
            throw new Error("Failed to acquire high-performance WebGPU adapter.");
        }

        // Query adapter info for telemetry HUD
        let gpuDesc = "WebGPU Accelerated Device";
        try {
            if (this.adapter.info) {
                const info = this.adapter.info;
                gpuDesc = `${info.vendor || 'GPU'} ${info.architecture || ''} ${info.description || ''}`.trim();
            }
        } catch (_) {}

        this.device = await this.adapter.requestDevice();
        this.context = canvas.getContext('webgpu');
        this.format = navigator.gpu.getPreferredCanvasFormat();

        this.context.configure({
            device: this.device,
            format: this.format,
            alphaMode: 'opaque'
        });

        await this._initPipelines();
        await this.setResolution(this.width, this.height);

        return {
            gpuDesc,
            limits: this.device.limits
        };
    }

    async _initPipelines() {
        const computeModule = this.device.createShaderModule({
            label: 'NCA Compute Shader',
            code: COMPUTE_SHADER
        });

        const renderModule = this.device.createShaderModule({
            label: 'NCA Render Blit Shader',
            code: RENDER_SHADER
        });

        // Compute Pipeline Layout & Bind Group Layout
        const computeBindGroupLayout = this.device.createBindGroupLayout({
            label: 'Compute Bind Group Layout',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }
            ]
        });

        this.computePipeline = this.device.createComputePipeline({
            label: 'NCA Compute Pipeline',
            layout: this.device.createPipelineLayout({
                bindGroupLayouts: [computeBindGroupLayout]
            }),
            compute: {
                module: computeModule,
                entryPoint: 'main'
            }
        });
        this.computeBindGroupLayout = computeBindGroupLayout;

        // Render Pipeline Layout & Bind Group Layout
        const renderBindGroupLayout = this.device.createBindGroupLayout({
            label: 'Render Bind Group Layout',
            entries: [
                { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
                { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }
            ]
        });

        this.renderPipeline = this.device.createRenderPipeline({
            label: 'NCA Render Pipeline',
            layout: this.device.createPipelineLayout({
                bindGroupLayouts: [renderBindGroupLayout]
            }),
            vertex: {
                module: renderModule,
                entryPoint: 'vs_main'
            },
            fragment: {
                module: renderModule,
                entryPoint: 'fs_main',
                targets: [{ format: this.format }]
            },
            primitive: {
                topology: 'triangle-list'
            }
        });
        this.renderBindGroupLayout = renderBindGroupLayout;

        // Allocate uniform buffers (64 bytes aligned)
        this.simParamsBuffer = this.device.createBuffer({
            size: 64,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        this.renderParamsBuffer = this.device.createBuffer({
            size: 64,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        // Weights buffer: 8336 floats x 4 bytes = 33344 bytes (padded to 33792)
        this.weightsBuffer = this.device.createBuffer({
            size: 33792,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
    }

    async setResolution(width, height) {
        this.width = width;
        this.height = height;

        const cellCount = width * height;
        const bufferByteSize = cellCount * 16 * 4;

        if (this.bufferA) this.bufferA.destroy();
        if (this.bufferB) this.bufferB.destroy();
        if (this.stagingBuffer) this.stagingBuffer.destroy();

        this.bufferA = this.device.createBuffer({
            label: `State Buffer A (${width}x${height})`,
            size: bufferByteSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
        });

        this.bufferB = this.device.createBuffer({
            label: `State Buffer B (${width}x${height})`,
            size: bufferByteSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
        });

        // Staging buffer for async cell counting (reads 4th channel alpha)
        this.stagingBuffer = this.device.createBuffer({
            label: 'Staging Buffer for Cell Counting',
            size: 1024 * 4, // sample 1024 cells for statistical population telemetry
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
        });

        // Create ping-pong compute bind groups
        // Group 0: Read A -> Write B
        this.computeBindGroups[0] = this.device.createBindGroup({
            layout: this.computeBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.bufferA } },
                { binding: 1, resource: { buffer: this.bufferB } },
                { binding: 2, resource: { buffer: this.simParamsBuffer } },
                { binding: 3, resource: { buffer: this.weightsBuffer } }
            ]
        });

        // Group 1: Read B -> Write A
        this.computeBindGroups[1] = this.device.createBindGroup({
            layout: this.computeBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.bufferB } },
                { binding: 1, resource: { buffer: this.bufferA } },
                { binding: 2, resource: { buffer: this.simParamsBuffer } },
                { binding: 3, resource: { buffer: this.weightsBuffer } }
            ]
        });

        // Create render bind groups
        this.renderBindGroups[0] = this.device.createBindGroup({
            layout: this.renderBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.bufferA } },
                { binding: 1, resource: { buffer: this.renderParamsBuffer } }
            ]
        });

        this.renderBindGroups[1] = this.device.createBindGroup({
            layout: this.renderBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.bufferB } },
                { binding: 1, resource: { buffer: this.renderParamsBuffer } }
            ]
        });

        this.currentBufferIdx = 0;
    }

    loadModelWeights(packedWeights) {
        this.device.queue.writeBuffer(this.weightsBuffer, 0, packedWeights);
    }

    seed(seedType) {
        const cellCount = this.width * this.height;
        const initialData = new Float32Array(cellCount * 16);
        const cx = Math.floor(this.width / 2);
        const cy = Math.floor(this.height / 2);

        if (seedType === 'center_single') {
            // Distill standard seed: 3x3 embryo cluster at center
            for (let dy = -1; dy <= 1; dy++) {
                for (let dx = -1; dx <= 1; dx++) {
                    const idx = ((cy + dy) * this.width + (cx + dx)) * 16;
                    // RGB
                    initialData[idx + 0] = 0.2;
                    initialData[idx + 1] = 0.8;
                    initialData[idx + 2] = 0.4;
                    // Alpha (living mask)
                    initialData[idx + 3] = 1.0;
                    // Hidden channels 4..15
                    for (let c = 4; c < 16; c++) {
                        initialData[idx + c] = 1.0;
                    }
                }
            }
        } else if (seedType === 'slime_cluster') {
            // Slime Mold: Seed nodes & nutrient attractant patches
            for (let i = 0; i < 7; i++) {
                const angle = (i / 7) * Math.PI * 2;
                const r = this.width * 0.28;
                const nodeX = Math.floor(cx + Math.cos(angle) * r);
                const nodeY = Math.floor(cy + Math.sin(angle) * r);
                
                for (let dy = -6; dy <= 6; dy++) {
                    for (let dx = -6; dx <= 6; dx++) {
                        if (dx * dx + dy * dy <= 36) {
                            const idx = ((nodeY + dy) * this.width + (nodeX + dx)) * 16;
                            initialData[idx + 0] = 0.9;
                            initialData[idx + 1] = 0.95;
                            initialData[idx + 2] = 0.15;
                            initialData[idx + 3] = 1.0;
                            initialData[idx + 4] = 0.9; // High attractant
                            initialData[idx + 7] = 1.0; // Conductivity
                        }
                    }
                }
            }
            // Central feeder core
            for (let dy = -8; dy <= 8; dy++) {
                for (let dx = -8; dx <= 8; dx++) {
                    if (dx * dx + dy * dy <= 64) {
                        const idx = ((cy + dy) * this.width + (cx + dx)) * 16;
                        initialData[idx + 0] = 0.95;
                        initialData[idx + 1] = 0.98;
                        initialData[idx + 2] = 0.2;
                        initialData[idx + 3] = 1.0;
                        initialData[idx + 4] = 1.0;
                    }
                }
            }
        } else if (seedType === 'jelly_core') {
            // Bioluminescent Jellyfish core
            for (let dy = -12; dy <= 12; dy++) {
                for (let dx = -12; dx <= 12; dx++) {
                    const d2 = dx * dx + dy * dy;
                    if (d2 <= 144) {
                        const idx = ((cy + dy) * this.width + (cx + dx)) * 16;
                        const dist = Math.sqrt(d2);
                        initialData[idx + 0] = 0.05;
                        initialData[idx + 1] = 0.4;
                        initialData[idx + 2] = 0.9;
                        initialData[idx + 3] = 1.0 - dist / 12;
                        initialData[idx + 4] = dist / 12; // Radial phase
                        initialData[idx + 8] = 1.0; // Luminescence burst
                    }
                }
            }
        } else if (seedType === 'coral_spores') {
            // Alien Coral Biocrystal: Distributed crystalline seeds
            const sporePoints = [
                [cx, cy],
                [cx - 80, cy - 60],
                [cx + 80, cy - 60],
                [cx - 70, cy + 80],
                [cx + 70, cy + 80]
            ];
            for (const [sx, sy] of sporePoints) {
                for (let dy = -4; dy <= 4; dy++) {
                    for (let dx = -4; dx <= 4; dx++) {
                        if (dx * dx + dy * dy <= 16) {
                            const idx = ((sy + dy) * this.width + (sx + dx)) * 16;
                            initialData[idx + 0] = 0.8;
                            initialData[idx + 1] = 0.2;
                            initialData[idx + 2] = 0.9;
                            initialData[idx + 3] = 1.0;
                            initialData[idx + 4] = 1.0; // Nutrient
                            initialData[idx + 5] = Math.random() * Math.PI; // Random crystal orientation
                        }
                    }
                }
            }
        } else if (seedType === 'turing_spots') {
            // Turing Reaction-Diffusion: Homogeneous U=1.0 with perturbed V=1.0 patches
            for (let i = 0; i < cellCount; i++) {
                const idx = i * 16;
                initialData[idx + 0] = 1.0; // U
                initialData[idx + 1] = 0.0; // V
                initialData[idx + 3] = 1.0; // Viability
            }
            // Plant 20 seed perturbation discs
            for (let s = 0; s < 24; s++) {
                const rx = Math.floor(Math.random() * (this.width - 40)) + 20;
                const ry = Math.floor(Math.random() * (this.height - 40)) + 20;
                const r = Math.floor(Math.random() * 8) + 6;
                for (let dy = -r; dy <= r; dy++) {
                    for (let dx = -r; dx <= r; dx++) {
                        if (dx * dx + dy * dy <= r * r) {
                            const idx = ((ry + dy) * this.width + (rx + dx)) * 16;
                            initialData[idx + 0] = 0.5;
                            initialData[idx + 1] = 0.25;
                            initialData[idx + 2] = 0.8;
                        }
                    }
                }
            }
        } else if (seedType === 'clear') {
            // Completely blank grid
        }

        // Upload to both buffers
        this.device.queue.writeBuffer(this.bufferA, 0, initialData);
        this.device.queue.writeBuffer(this.bufferB, 0, initialData);
        this.stepCounter = 0;
    }

    step(simParams) {
        const t0 = performance.now();
        this.stepCounter++;

        // Pack SimParams uniform struct (16 x 4-byte values = 64 bytes)
        const paramsArray = new ArrayBuffer(64);
        const u32View = new Uint32Array(paramsArray);
        const f32View = new Float32Array(paramsArray);

        u32View[0] = this.width;
        u32View[1] = this.height;
        u32View[2] = this.stepCounter;
        f32View[3] = Math.random() * 1000.0; // seed

        u32View[4] = simParams.modelType;
        f32View[5] = simParams.updateRate;
        f32View[6] = simParams.aliveThreshold;
        f32View[7] = performance.now() * 0.001; // time in seconds

        u32View[8] = simParams.brushActive ? 1 : 0;
        u32View[9] = simParams.brushType;
        f32View[10] = simParams.brushX;
        f32View[11] = simParams.brushY;

        f32View[12] = simParams.brushRadius;
        f32View[13] = simParams.brushStrength;
        f32View[14] = simParams.mutationRate;
        f32View[15] = 0.0; // padding

        this.device.queue.writeBuffer(this.simParamsBuffer, 0, paramsArray);

        // Encode Compute Pass
        const commandEncoder = this.device.createCommandEncoder({ label: 'NCA Step Encoder' });
        const computePass = commandEncoder.beginComputePass({ label: 'NCA Compute Pass' });
        computePass.setPipeline(this.computePipeline);
        computePass.setBindGroup(0, this.computeBindGroups[this.currentBufferIdx]);

        const workgroupsX = Math.ceil(this.width / 16);
        const workgroupsY = Math.ceil(this.height / 16);
        computePass.dispatchWorkgroups(workgroupsX, workgroupsY);
        computePass.end();

        this.device.queue.submit([commandEncoder.finish()]);

        // Ping-pong flip: The output of this step becomes input for next step
        this.currentBufferIdx = 1 - this.currentBufferIdx;

        const latency = performance.now() - t0;
        return latency;
    }

    render(renderParams) {
        // Pack RenderParams uniform struct (16 x 4-byte values = 64 bytes)
        const paramsArray = new ArrayBuffer(64);
        const u32View = new Uint32Array(paramsArray);
        const f32View = new Float32Array(paramsArray);

        f32View[0] = this.width;
        f32View[1] = this.height;
        f32View[2] = renderParams.zoom;
        f32View[3] = renderParams.panX;

        f32View[4] = renderParams.panY;
        f32View[5] = this.canvas.width;
        f32View[6] = this.canvas.height;
        u32View[7] = renderParams.displayMode;

        u32View[8] = renderParams.selectedChannel;
        u32View[9] = renderParams.brushActive ? 1 : 0;
        f32View[10] = renderParams.brushX;
        f32View[11] = renderParams.brushY;

        f32View[12] = renderParams.brushRadius;
        u32View[13] = renderParams.brushType;
        f32View[14] = performance.now() * 0.001;
        f32View[15] = 0.0;

        this.device.queue.writeBuffer(this.renderParamsBuffer, 0, paramsArray);

        const currentTexture = this.context.getCurrentTexture();
        const commandEncoder = this.device.createCommandEncoder({ label: 'NCA Render Encoder' });
        const renderPass = commandEncoder.beginRenderPass({
            label: 'NCA Canvas Blit Pass',
            colorAttachments: [{
                view: currentTexture.createView(),
                clearValue: { r: 0.03, g: 0.04, b: 0.06, a: 1.0 },
                loadOp: 'clear',
                storeOp: 'store'
            }]
        });

        renderPass.setPipeline(this.renderPipeline);
        // Bind the buffer that holds the most recent state
        renderPass.setBindGroup(0, this.renderBindGroups[this.currentBufferIdx]);
        renderPass.draw(3); // Fullscreen triangle
        renderPass.end();

        this.device.queue.submit([commandEncoder.finish()]);
    }

    async sampleLivingCellsAsync() {
        if (this.isStagingBusy || !this.stagingBuffer) return this.lastLivingCount;
        this.isStagingBusy = true;

        try {
            const encoder = this.device.createCommandEncoder();
            const sourceBuffer = (this.currentBufferIdx === 0) ? this.bufferA : this.bufferB;
            encoder.copyBufferToBuffer(sourceBuffer, 0, this.stagingBuffer, 0, 1024 * 4);
            this.device.queue.submit([encoder.finish()]);

            await this.stagingBuffer.mapAsync(GPUMapMode.READ);
            const copyArray = new Float32Array(this.stagingBuffer.getMappedRange());
            let livingSample = 0;
            const sampleSize = copyArray.length / 16;
            for (let i = 0; i < sampleSize; i++) {
                if (copyArray[i * 16 + 3] > 0.1) {
                    livingSample++;
                }
            }
            this.stagingBuffer.unmap();

            // Extrapolate population estimate
            const ratio = livingSample / sampleSize;
            this.lastLivingCount = Math.round(ratio * (this.width * this.height));
        } catch (_) {
            // Ignore staging contention
        } finally {
            this.isStagingBusy = false;
        }

        return this.lastLivingCount;
    }
}
