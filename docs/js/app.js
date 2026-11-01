// Main Application Coordinator
// Orchestrates simulation loop, WebGPU/WebGL engine management, UI controls, and telemetry.

import { WebGPUEngine } from './webgpu-engine.js';
import { WebGLFallbackEngine } from './webgl-engine.js';
import { PRETRAINED_MODELS, unpackModelWeights, ECOSYSTEM_PRESETS } from './models.js';
import { TelemetryMonitor } from './telemetry.js';

class NeuralAutomataApp {
    constructor() {
        this.canvas = document.getElementById('sim-canvas');
        this.engine = null;
        this.isWebGPU = false;
        this.telemetry = new TelemetryMonitor();
        
        // Simulation State
        this.isRunning = true;
        this.stepsPerFrame = 4;
        this.currentPreset = ECOSYSTEM_PRESETS[0]; // Gecko
        this.activeWeights = null;
        this.originalWeights = null;
        
        this.simParams = {
            modelType: 0,
            updateRate: 0.5,
            aliveThreshold: 0.1,
            brushActive: false,
            brushType: 0, // 0=Scalpel, 1=Seed, 2=Mutator, 3=Nutrient
            brushX: 0,
            brushY: 0,
            brushRadius: 18,
            brushStrength: 1.0,
            mutationRate: 0.0
        };

        // Navigation (Zoom & Pan)
        this.renderParams = {
            zoom: 1.0,
            panX: 0.0,
            panY: 0.0,
            displayMode: 0, // 0=RGBA, 1=Hidden Channel, 2=Alpha, 3=Gradients
            selectedChannel: 4,
            brushActive: false,
            brushX: 0,
            brushY: 0,
            brushRadius: 18,
            brushType: 0
        };

        this.isDragging = false;
        this.isPanning = false;
        this.lastMouseX = 0;
        this.lastMouseY = 0;
        
        this.ui = {};
    }

    async start() {
        this._bindUIElements();
        this._resizeCanvas();
        window.addEventListener('resize', () => this._resizeCanvas());

        // Initialize Engine: try WebGPU first, fallback to WebGL 2.0
        try {
            const gpuEngine = new WebGPUEngine();
            const info = await gpuEngine.init(this.canvas);
            this.engine = gpuEngine;
            this.isWebGPU = true;
            this.telemetry.setEngineMode("WebGPU Active (Hardware Accelerated)");
            this.telemetry.setGpuInfo(info.gpuDesc);
            console.log("WebGPU initialized successfully:", info);
        } catch (err) {
            console.warn("WebGPU initialization failed, engaging WebGL 2.0 fallback:", err);
            const fallbackEngine = new WebGLFallbackEngine();
            const info = await fallbackEngine.init(this.canvas);
            this.engine = fallbackEngine;
            this.isWebGPU = false;
            this.telemetry.setEngineMode("WebGL 2.0 Active (Fallback Engine)");
            this.telemetry.setGpuInfo(info.gpuDesc);
        }

        this.telemetry.setGridResolution(this.engine.width, this.engine.height);
        this._populatePresetMenu();
        this._bindInputHandlers();
        this._bindControlHandlers();

        // Load default preset (Gecko)
        await this.loadPreset(this.currentPreset);

        // Start animation loop
        requestAnimationFrame((t) => this._loop(t));

        // Periodic async cell sampling for telemetry HUD (every 300ms)
        setInterval(async () => {
            if (this.engine) {
                const count = await this.engine.sampleLivingCellsAsync();
                this.telemetry.updateLivingCount(count);
            }
        }, 300);
    }

    _bindUIElements() {
        this.ui = {
            fps: document.getElementById('val-fps'),
            latency: document.getElementById('val-latency'),
            cells: document.getElementById('val-cells'),
            memory: document.getElementById('val-memory'),
            gpu: document.getElementById('val-gpu'),
            engineBadge: document.getElementById('engine-badge'),
            stepCount: document.getElementById('val-steps'),
            
            btnPlayPause: document.getElementById('btn-play-pause'),
            btnStep: document.getElementById('btn-step'),
            btnReset: document.getElementById('btn-reset'),
            btnClear: document.getElementById('btn-clear'),
            
            resolutionSelect: document.getElementById('select-resolution'),
            speedSelect: document.getElementById('select-speed'),
            displayModeSelect: document.getElementById('select-display-mode'),
            channelSlider: document.getElementById('slider-channel'),
            channelVal: document.getElementById('val-channel'),
            
            brushRadiuSlider: document.getElementById('slider-brush-radius'),
            brushRadiusVal: document.getElementById('val-brush-radius'),
            updateRateSlider: document.getElementById('slider-update-rate'),
            updateRateVal: document.getElementById('val-update-rate'),
            
            mutationSlider: document.getElementById('slider-mutation'),
            mutationVal: document.getElementById('val-mutation'),
            btnMutateNow: document.getElementById('btn-mutate-now'),
            btnResetWeights: document.getElementById('btn-reset-weights'),
            
            presetContainer: document.getElementById('preset-cards-container'),
            presetName: document.getElementById('preset-name-display'),
            presetDesc: document.getElementById('preset-desc-display'),
            btnResetView: document.getElementById('btn-reset-view'),
            
            toolButtons: document.querySelectorAll('.tool-btn')
        };
    }

    _resizeCanvas() {
        const container = this.canvas.parentElement;
        const rect = container.getBoundingClientRect();
        const dpr = Math.min(window.devicePixelRatio || 1, 2);
        
        const targetW = Math.floor(rect.width * dpr);
        const targetH = Math.floor(rect.height * dpr);
        
        if (this.canvas.width !== targetW || this.canvas.height !== targetH) {
            this.canvas.width = targetW;
            this.canvas.height = targetH;
            if (this.isWebGPU && this.engine && this.engine.context) {
                this.engine.context.configure({
                    device: this.engine.device,
                    format: this.engine.format,
                    alphaMode: 'opaque'
                });
            }
        }
    }

    _populatePresetMenu() {
        if (!this.ui.presetContainer) return;
        this.ui.presetContainer.innerHTML = '';

        ECOSYSTEM_PRESETS.forEach(preset => {
            const card = document.createElement('button');
            card.className = `preset-card ${preset.id === this.currentPreset.id ? 'active' : ''}`;
            card.dataset.presetId = preset.id;
            card.innerHTML = `
                <div class="preset-header">
                    <span class="preset-title">${preset.name}</span>
                    <span class="preset-badge" style="background: ${preset.colorAccent}22; color: ${preset.colorAccent}; border: 1px solid ${preset.colorAccent}44;">${preset.badge}</span>
                </div>
                <div class="preset-sub">${preset.subtitle}</div>
            `;
            card.addEventListener('click', () => {
                this.loadPreset(preset);
            });
            this.ui.presetContainer.appendChild(card);
        });
    }

    async loadPreset(preset) {
        this.currentPreset = preset;
        this.simParams.modelType = preset.modelType;
        this.stepsPerFrame = preset.speed || 4;
        if (this.ui.speedSelect) this.ui.speedSelect.value = this.stepsPerFrame;

        // Update active UI card
        const cards = document.querySelectorAll('.preset-card');
        cards.forEach(c => {
            c.classList.toggle('active', c.dataset.presetId === preset.id);
        });

        if (this.ui.presetName) this.ui.presetName.textContent = preset.name;
        if (this.ui.presetDesc) this.ui.presetDesc.textContent = preset.desc;

        // If Neural model, load weights
        if (preset.modelKey && PRETRAINED_MODELS[preset.modelKey]) {
            const unpacked = unpackModelWeights(PRETRAINED_MODELS[preset.modelKey]);
            this.originalWeights = new Float32Array(unpacked.packed);
            this.activeWeights = new Float32Array(unpacked.packed);
            this.engine.loadModelWeights(this.activeWeights);
        }

        // Seed initial state
        this.engine.seed(preset.seedType);
        
        // Reset view center
        this._resetView();
    }

    _resetView() {
        this.renderParams.zoom = Math.min(
            (this.canvas.width * 0.8) / this.engine.width,
            (this.canvas.height * 0.8) / this.engine.height
        );
        this.renderParams.panX = 0;
        this.renderParams.panY = 0;
    }

    _bindControlHandlers() {
        // Play / Pause
        if (this.ui.btnPlayPause) {
            this.ui.btnPlayPause.addEventListener('click', () => {
                this.isRunning = !this.isRunning;
                this.ui.btnPlayPause.innerHTML = this.isRunning
                    ? `<svg class="icon" viewBox="0 0 24 24"><path fill="currentColor" d="M14 19h4V5h-4v14zm-8 0h4V5H6v14z"/></svg> PAUSE`
                    : `<svg class="icon" viewBox="0 0 24 24"><path fill="currentColor" d="M8 5v14l11-7z"/></svg> RUN`;
                this.ui.btnPlayPause.classList.toggle('active-run', this.isRunning);
            });
        }

        // Step 1 Frame
        if (this.ui.btnStep) {
            this.ui.btnStep.addEventListener('click', () => {
                this.isRunning = false;
                if (this.ui.btnPlayPause) {
                    this.ui.btnPlayPause.innerHTML = `<svg class="icon" viewBox="0 0 24 24"><path fill="currentColor" d="M8 5v14l11-7z"/></svg> RUN`;
                    this.ui.btnPlayPause.classList.remove('active-run');
                }
                const lat = this.engine.step(this.simParams);
                this.telemetry.recordComputeStep(lat);
            });
        }

        // Reset Seed
        if (this.ui.btnReset) {
            this.ui.btnReset.addEventListener('click', () => {
                this.engine.seed(this.currentPreset.seedType);
            });
        }

        // Clear Grid
        if (this.ui.btnClear) {
            this.ui.btnClear.addEventListener('click', () => {
                this.engine.seed('clear');
            });
        }

        // Reset View
        if (this.ui.btnResetView) {
            this.ui.btnResetView.addEventListener('click', () => this._resetView());
        }

        // Resolution Select
        if (this.ui.resolutionSelect) {
            this.ui.resolutionSelect.addEventListener('change', async (e) => {
                const res = parseInt(e.target.value, 10);
                await this.engine.setResolution(res, res);
                this.telemetry.setGridResolution(res, res);
                await this.loadPreset(this.currentPreset);
            });
        }

        // Speed Multiplier
        if (this.ui.speedSelect) {
            this.ui.speedSelect.addEventListener('change', (e) => {
                this.stepsPerFrame = parseInt(e.target.value, 10);
            });
        }

        // Display Mode Select
        if (this.ui.displayModeSelect) {
            this.ui.displayModeSelect.addEventListener('change', (e) => {
                this.renderParams.displayMode = parseInt(e.target.value, 10);
                const channelSliderGroup = document.getElementById('group-channel-slider');
                if (channelSliderGroup) {
                    channelSliderGroup.style.display = (this.renderParams.displayMode === 1) ? 'flex' : 'none';
                }
            });
        }

        // Hidden Channel Slider
        if (this.ui.channelSlider) {
            this.ui.channelSlider.addEventListener('input', (e) => {
                const ch = parseInt(e.target.value, 10);
                this.renderParams.selectedChannel = ch;
                if (this.ui.channelVal) this.ui.channelVal.textContent = `Channel ${ch}`;
            });
        }

        // Brush Radius Slider
        if (this.ui.brushRadiuSlider) {
            this.ui.brushRadiuSlider.addEventListener('input', (e) => {
                const r = parseInt(e.target.value, 10);
                this.simParams.brushRadius = r;
                this.renderParams.brushRadius = r;
                if (this.ui.brushRadiusVal) this.ui.brushRadiusVal.textContent = `${r} px`;
            });
        }

        // Update Rate Slider
        if (this.ui.updateRateSlider) {
            this.ui.updateRateSlider.addEventListener('input', (e) => {
                const rate = parseFloat(e.target.value);
                this.simParams.updateRate = rate;
                if (this.ui.updateRateVal) this.ui.updateRateVal.textContent = `${Math.round(rate * 100)}%`;
            });
        }

        // Tool Selection Buttons
        this.ui.toolButtons.forEach(btn => {
            btn.addEventListener('click', () => {
                this.ui.toolButtons.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                const toolType = parseInt(btn.dataset.tool, 10);
                this.simParams.brushType = toolType;
                this.renderParams.brushType = toolType;
            });
        });

        // Mutation Slider & Mutator Lab
        if (this.ui.mutationSlider) {
            this.ui.mutationSlider.addEventListener('input', (e) => {
                const mag = parseFloat(e.target.value);
                if (this.ui.mutationVal) this.ui.mutationVal.textContent = `${mag.toFixed(2)}`;
            });
        }

        if (this.ui.btnMutateNow) {
            this.ui.btnMutateNow.addEventListener('click', () => {
                const mag = parseFloat(this.ui.mutationSlider.value || 0.15);
                this._applyWeightMutation(mag);
            });
        }

        if (this.ui.btnResetWeights) {
            this.ui.btnResetWeights.addEventListener('click', () => {
                if (this.originalWeights && this.activeWeights) {
                    this.activeWeights.set(this.originalWeights);
                    this.engine.loadModelWeights(this.activeWeights);
                }
            });
        }
    }

    _applyWeightMutation(magnitude) {
        if (!this.activeWeights || !this.originalWeights) return;
        for (let i = 0; i < this.activeWeights.length; i++) {
            const delta = (Math.random() * 2 - 1) * magnitude * Math.abs(this.originalWeights[i] + 0.05);
            this.activeWeights[i] += delta;
        }
        this.engine.loadModelWeights(this.activeWeights);
    }

    _bindInputHandlers() {
        const getGridCoords = (clientX, clientY) => {
            const rect = this.canvas.getBoundingClientRect();
            const dpr = Math.min(window.devicePixelRatio || 1, 2);
            const canvasX = (clientX - rect.left) * dpr;
            const canvasY = (clientY - rect.top) * dpr;

            const centerCanvasX = this.canvas.width * 0.5;
            const centerCanvasY = this.canvas.height * 0.5;

            const gx = (canvasX - centerCanvasX - this.renderParams.panX) / this.renderParams.zoom + this.engine.width * 0.5;
            const gy = (canvasY - centerCanvasY - this.renderParams.panY) / this.renderParams.zoom + this.engine.height * 0.5;

            return { gx, gy, canvasX, canvasY };
        };

        // Pointer Down
        this.canvas.addEventListener('pointerdown', (e) => {
            this.canvas.setPointerCapture(e.pointerId);
            this.lastMouseX = e.clientX;
            this.lastMouseY = e.clientY;

            if (e.button === 2 || e.shiftKey || e.altKey) {
                // Right-click or Shift-click = Pan
                this.isPanning = true;
            } else {
                // Left-click = Brush Tool
                this.isDragging = true;
                const coords = getGridCoords(e.clientX, e.clientY);
                this.simParams.brushActive = true;
                this.simParams.brushX = coords.gx;
                this.simParams.brushY = coords.gy;
                this.renderParams.brushActive = true;
                this.renderParams.brushX = coords.gx;
                this.renderParams.brushY = coords.gy;
            }
        });

        // Pointer Move
        this.canvas.addEventListener('pointermove', (e) => {
            const coords = getGridCoords(e.clientX, e.clientY);
            this.renderParams.brushX = coords.gx;
            this.renderParams.brushY = coords.gy;

            if (this.isPanning) {
                const dx = (e.clientX - this.lastMouseX) * Math.min(window.devicePixelRatio || 1, 2);
                const dy = (e.clientY - this.lastMouseY) * Math.min(window.devicePixelRatio || 1, 2);
                this.renderParams.panX += dx;
                this.renderParams.panY += dy;
                this.lastMouseX = e.clientX;
                this.lastMouseY = e.clientY;
            } else if (this.isDragging) {
                this.simParams.brushActive = true;
                this.simParams.brushX = coords.gx;
                this.simParams.brushY = coords.gy;
            }
        });

        // Pointer Up / Cancel
        const endPointer = (e) => {
            try { this.canvas.releasePointerCapture(e.pointerId); } catch (_) {}
            this.isDragging = false;
            this.isPanning = false;
            this.simParams.brushActive = false;
        };
        this.canvas.addEventListener('pointerup', endPointer);
        this.canvas.addEventListener('pointercancel', endPointer);

        // Hover tracking for tool reticle
        this.canvas.addEventListener('pointerenter', () => {
            this.renderParams.brushActive = true;
        });
        this.canvas.addEventListener('pointerleave', () => {
            this.renderParams.brushActive = false;
            this.simParams.brushActive = false;
        });

        // Prevent Context Menu on canvas
        this.canvas.addEventListener('contextmenu', (e) => e.preventDefault());

        // Mouse Wheel Zoom
        this.canvas.addEventListener('wheel', (e) => {
            e.preventDefault();
            const zoomFactor = e.deltaY < 0 ? 1.15 : 0.87;
            const newZoom = Math.max(0.2, Math.min(30.0, this.renderParams.zoom * zoomFactor));

            // Zoom toward cursor
            const rect = this.canvas.getBoundingClientRect();
            const dpr = Math.min(window.devicePixelRatio || 1, 2);
            const mouseCanvasX = (e.clientX - rect.left) * dpr;
            const mouseCanvasY = (e.clientY - rect.top) * dpr;
            const centerCanvasX = this.canvas.width * 0.5;
            const centerCanvasY = this.canvas.height * 0.5;

            const preX = (mouseCanvasX - centerCanvasX - this.renderParams.panX) / this.renderParams.zoom;
            const preY = (mouseCanvasY - centerCanvasY - this.renderParams.panY) / this.renderParams.zoom;

            this.renderParams.zoom = newZoom;

            this.renderParams.panX = mouseCanvasX - centerCanvasX - preX * newZoom;
            this.renderParams.panY = mouseCanvasY - centerCanvasY - preY * newZoom;
        }, { passive: false });
    }

    _loop(timestamp) {
        this.telemetry.recordFrame(timestamp);

        // Step simulation
        if (this.isRunning && this.engine) {
            let totalLatency = 0;
            for (let i = 0; i < this.stepsPerFrame; i++) {
                const lat = this.engine.step(this.simParams);
                totalLatency += lat;
            }
            this.telemetry.recordComputeStep(totalLatency / this.stepsPerFrame);
        }

        // Render to canvas
        if (this.engine) {
            this.engine.render(this.renderParams);
        }

        // Update telemetry HUD
        this.telemetry.renderToUI(this.ui);

        requestAnimationFrame((t) => this._loop(t));
    }
}

// Bootstrap on DOM load
window.addEventListener('DOMContentLoaded', () => {
    const app = new NeuralAutomataApp();
    app.start();
    window.NCA_APP = app; // Expose for testing/debugging
});
