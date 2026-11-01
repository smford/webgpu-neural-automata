// Performance Telemetry & Hardware Monitor
// Real-time tracking of FPS, compute pass latency, active living cells, and memory footprint.

export class TelemetryMonitor {
    constructor() {
        this.fps = 0;
        this.frameCount = 0;
        this.lastTime = performance.now();
        this.fpsHistory = new Float32Array(60);
        this.fpsHistoryIdx = 0;
        
        this.computeLatency = 0;
        this.computeHistory = new Float32Array(60);
        this.computeHistoryIdx = 0;
        
        this.livingCells = 0;
        this.livingPercent = 0;
        this.totalCells = 0;
        
        this.memoryMB = 0;
        this.gpuInfo = "Detecting GPU...";
        this.engineMode = "Initializing...";
        
        this.stepCounter = 0;
        this.lastUiUpdate = 0;
    }

    setGpuInfo(info) {
        this.gpuInfo = info;
    }

    setEngineMode(mode) {
        this.engineMode = mode;
    }

    setGridResolution(width, height) {
        this.totalCells = width * height;
        // Ping-pong 16 channels x 4 bytes x 2 buffers + weights
        const bufferBytes = (width * height * 16 * 4) * 2;
        const weightsBytes = 8336 * 4;
        this.memoryMB = ((bufferBytes + weightsBytes) / (1024 * 1024)).toFixed(1);
    }

    recordComputeStep(latencyMs) {
        this.computeLatency = this.computeLatency * 0.85 + latencyMs * 0.15;
        this.computeHistory[this.computeHistoryIdx] = latencyMs;
        this.computeHistoryIdx = (this.computeHistoryIdx + 1) % 60;
        this.stepCounter++;
    }

    recordFrame(now) {
        this.frameCount++;
        const delta = now - this.lastTime;
        if (delta >= 250) {
            this.fps = (this.frameCount * 1000) / delta;
            this.fpsHistory[this.fpsHistoryIdx] = this.fps;
            this.fpsHistoryIdx = (this.fpsHistoryIdx + 1) % 60;
            this.frameCount = 0;
            this.lastTime = now;
        }
    }

    updateLivingCount(count) {
        this.livingCells = count;
        if (this.totalCells > 0) {
            this.livingPercent = ((count / this.totalCells) * 100).toFixed(2);
        }
    }

    renderToUI(elements) {
        const now = performance.now();
        if (now - this.lastUiUpdate < 80) return; // Cap DOM updates to ~12Hz for zero UI jank
        this.lastUiUpdate = now;

        if (elements.fps) {
            elements.fps.textContent = Math.round(this.fps);
        }
        if (elements.latency) {
            elements.latency.textContent = this.computeLatency.toFixed(2);
        }
        if (elements.cells) {
            elements.cells.textContent = `${this.livingCells.toLocaleString()} (${this.livingPercent}%)`;
        }
        if (elements.memory) {
            elements.memory.textContent = `${this.memoryMB} MB`;
        }
        if (elements.gpu) {
            elements.gpu.textContent = this.gpuInfo;
        }
        if (elements.engineBadge) {
            elements.engineBadge.textContent = this.engineMode;
        }
        if (elements.stepCount) {
            elements.stepCount.textContent = this.stepCounter.toLocaleString();
        }
    }
}
