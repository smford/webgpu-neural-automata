// WebGPU Shading Language (WGSL) Compute & Render Shaders
// High-performance Neural Cellular Automata simulation and blit pipeline.

export const COMPUTE_SHADER = `
struct SimParams {
    width: u32,
    height: u32,
    step: u32,
    seed: f32,
    
    model_type: u32,        // 0=Neural NCA, 1=Slime Mold, 2=Jellyfish, 3=Biocrystal, 4=Turing RD
    update_rate: f32,       // Stochastic update probability (default 0.5)
    alive_threshold: f32,   // Alpha threshold (default 0.1)
    time: f32,
    
    brush_active: u32,      // 1 if brush currently interacting, 0 otherwise
    brush_type: u32,        // 0=Scalpel/Eraser, 1=Seed Pen, 2=Mutator, 3=Nutrient
    brush_x: f32,           // Grid coordinate X
    brush_y: f32,           // Grid coordinate Y
    
    brush_radius: f32,      // Brush radius in cell units
    brush_strength: f32,    // Brush intensity multiplier
    mutation_rate: f32,     // Live weight perturbation scale
    padding: f32,
};

@group(0) @binding(0) var<storage, read> stateIn: array<f32>;
@group(0) @binding(1) var<storage, read_write> stateOut: array<f32>;
@group(0) @binding(2) var<uniform> params: SimParams;
@group(0) @binding(3) var<storage, read> weights: array<f32>;

// Pseudo-random hash without sine (Hoskins / Dave_Hoskins)
fn hash13(p3_in: vec3<f32>) -> f32 {
    var p3 = fract(p3_in * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

fn get_sobel_x(dx: i32, dy: i32) -> f32 {
    var sx = 0.0;
    if (dx == -1) { sx = -1.0; }
    else if (dx == 1) { sx = 1.0; }
    if (dy == 0) { sx *= 2.0; }
    return sx * 0.125;
}

fn get_sobel_y(dx: i32, dy: i32) -> f32 {
    var sy = 0.0;
    if (dy == -1) { sy = -1.0; }
    else if (dy == 1) { sy = 1.0; }
    if (dx == 0) { sy *= 2.0; }
    return sy * 0.125;
}

fn get_laplacian(dx: i32, dy: i32) -> f32 {
    if (dx == 0 && dy == 0) { return -1.5; }
    if (dx == 0 || dy == 0) { return 0.25; }
    return 0.125;
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = id.x;
    let y = id.y;
    
    if (x >= params.width || y >= params.height) {
        return;
    }
    
    let current_idx = (y * params.width + x) * 16u;
    let w_i32 = i32(params.width);
    let h_i32 = i32(params.height);
    
    // Convolve 3x3 neighborhood across 16 channels:
    // Identity (16), Sobel-X (16), Sobel-Y (16), Laplacian (16)
    var p: array<f32, 48>;
    var lap: array<f32, 16>;
    for (var i = 0u; i < 48u; i++) { p[i] = 0.0; }
    for (var i = 0u; i < 16u; i++) { lap[i] = 0.0; }
    
    var max_alpha = 0.0;
    var center_state: array<f32, 16>;
    
    for (var dy = -1; dy <= 1; dy++) {
        let ny = (i32(y) + dy + h_i32) % h_i32;
        for (var dx = -1; dx <= 1; dx++) {
            let nx = (i32(x) + dx + w_i32) % w_i32;
            let n_idx = (u32(ny) * params.width + u32(nx)) * 16u;
            
            let sx = get_sobel_x(dx, dy);
            let sy = get_sobel_y(dx, dy);
            let lp = get_laplacian(dx, dy);
            
            let neighbor_alpha = stateIn[n_idx + 3u];
            max_alpha = max(max_alpha, neighbor_alpha);
            
            let is_center = (dx == 0 && dy == 0);
            for (var c = 0u; c < 16u; c++) {
                let val = stateIn[n_idx + c];
                if (is_center) {
                    p[c] = val;
                    center_state[c] = val;
                }
                p[16u + c] += sx * val;
                p[32u + c] += sy * val;
                lap[c] += lp * val;
            }
        }
    }
    
    var new_state: array<f32, 16>;
    for (var c = 0u; c < 16u; c++) {
        new_state[c] = center_state[c];
    }
    
    // Dispatch according to selected model architecture
    if (params.model_type == 0u) {
        // ==========================================
        // 1. NEURAL NCA (2-Layer Perceptron MLP)
        // ==========================================
        // Alive Masking Check: If cell and all 8 neighbors are dead (alpha <= threshold), stay dead!
        if (max_alpha <= params.alive_threshold) {
            for (var c = 0u; c < 16u; c++) {
                stateOut[current_idx + c] = 0.0;
            }
            return;
        }
        
        // Layer 1: 48 Inputs -> 128 Hidden Neurons with ReLU activation
        // Weights layout: W1 [0..6143], B1 [6144..6271]
        var hidden: array<f32, 128>;
        for (var j = 0u; j < 128u; j++) {
            var sum = weights[6144u + j]; // Bias B1[j]
            for (var k = 0u; k < 48u; k++) {
                sum += p[k] * weights[k * 128u + j];
            }
            hidden[j] = max(0.0, sum); // ReLU
        }
        
        // Layer 2: 128 Hidden -> 16 Output Deltas (Linear)
        // Weights layout: W2 [6272..8319], B2 [8320..8335]
        var delta: array<f32, 16>;
        for (var c = 0u; c < 16u; c++) {
            var sum = weights[8320u + c]; // Bias B2[c]
            for (var j = 0u; j < 128u; j++) {
                sum += hidden[j] * weights[6272u + j * 16u + c];
            }
            delta[c] = sum;
        }
        
        // Stochastic Update Masking: update random subset (~50%) to emulate asynchronous cellular biology
        let rand_val = hash13(vec3<f32>(f32(x), f32(y), params.seed));
        let should_update = rand_val < params.update_rate;
        
        for (var c = 0u; c < 16u; c++) {
            if (should_update) {
                new_state[c] = center_state[c] + delta[c];
            }
        }
        
        // Post-alive verification: if alpha <= threshold and no living neighbor, cell ceases to exist
        if (new_state[3] <= params.alive_threshold && max_alpha <= params.alive_threshold) {
            for (var c = 0u; c < 16u; c++) {
                stateOut[current_idx + c] = 0.0;
            }
            return;
        }
        
    } else if (params.model_type == 1u) {
        // ==========================================
        // 2. PHYSARUM SLIME MOLD (Chemotaxis & Vein Flow)
        // ==========================================
        let density = center_state[0];
        let attractant = center_state[4];
        let vx = center_state[5];
        let vy = center_state[6];
        let conductivity = center_state[7];
        
        let grad_a_x = p[16u + 4u];
        let grad_a_y = p[32u + 4u];
        let lap_d = lap[0];
        let lap_a = lap[4];
        
        // Shuttle streaming velocity aligned with chemo-gradient
        let flux_x = mix(vx, grad_a_x * 4.0, 0.25);
        let flux_y = mix(vy, grad_a_y * 4.0, 0.25);
        let flux_mag = sqrt(flux_x * flux_x + flux_y * flux_y);
        
        // Vein reinforcement where flux is strong, decay elsewhere
        let new_cond = clamp(conductivity * 0.98 + flux_mag * 0.08, 0.05, 2.0);
        
        // Mass transport
        let new_density = clamp(density + lap_d * (0.15 + new_cond * 0.2) + (flux_mag - density * 0.02) * 0.05, 0.0, 1.0);
        let new_attractant = clamp(attractant + lap_a * 0.2 - attractant * 0.005 + sin(params.time * 2.0 + f32(x + y) * 0.05) * 0.002, 0.0, 1.0);
        
        // Slime fluorescent color synthesis (vibrant lime yellow/green protoplasm)
        new_state[0] = mix(0.1, 0.92, new_density);
        new_state[1] = mix(0.15, 0.98, new_density);
        new_state[2] = mix(0.05, 0.20, new_cond * 0.5);
        new_state[3] = clamp(new_density * 1.5, 0.0, 1.0);
        new_state[4] = new_attractant;
        new_state[5] = flux_x;
        new_state[6] = flux_y;
        new_state[7] = new_cond;
        
    } else if (params.model_type == 2u) {
        // ==========================================
        // 3. BIOLUMINESCENT JELLYFISH (Pulsing Hydrozoa)
        // ==========================================
        let phase = center_state[4];
        let osc_speed = 0.08;
        let lap_phase = lap[4];
        
        // Radial distance from center for bell envelope
        let cx = f32(params.width) * 0.5;
        let cy = f32(params.height) * 0.5;
        let r_dist = distance(vec2<f32>(f32(x), f32(y)), vec2<f32>(cx, cy));
        let max_r = f32(params.width) * 0.35;
        
        // Pacemaker pulse at apex propagating outward
        let new_phase = fract(phase + osc_speed + lap_phase * 0.1);
        let pulse = pow(sin(new_phase * 6.2831853), 4.0);
        
        // Hydrodynamic bell deformation
        let contraction = sin(new_phase * 6.2831853 - r_dist * 0.08);
        let bell_edge = smoothstep(max_r * (0.8 + 0.15 * contraction), max_r * 0.3, r_dist);
        
        // Bioluminescent electric cyan / neon indigo emission
        let glow = clamp(pulse * bell_edge * 1.8 + center_state[8] * 0.9, 0.0, 1.0);
        new_state[0] = mix(0.02, 0.1, glow) + glow * 0.3;
        new_state[1] = mix(0.1, 0.85, glow);
        new_state[2] = mix(0.3, 1.0, glow);
        new_state[3] = clamp(bell_edge * (0.3 + glow * 0.7), 0.0, 1.0);
        new_state[4] = new_phase;
        new_state[8] = max(0.0, center_state[8] * 0.92); // Luminescence decay
        
    } else if (params.model_type == 3u) {
        // ==========================================
        // 4. ALIEN CORAL BIOCRYSTAL (Dendritic Mineral Growth)
        // ==========================================
        let crystal = center_state[0];
        let nutrient = center_state[4];
        let orientation = center_state[5];
        
        let lap_c = lap[0];
        let lap_n = lap[4];
        
        // Anisotropic 6-fold symmetry for hexagonal crystal facets
        let gx = p[16u + 0u];
        let gy = p[32u + 0u];
        let theta = atan2(gy, gx);
        let anisotropy = 1.0 + 0.35 * cos(6.0 * (theta + orientation));
        
        // Dendritic crystallization rate
        let growth = nutrient * (lap_c * 0.2 + crystal * (1.0 - crystal) * 0.8) * anisotropy;
        let new_crystal = clamp(crystal + growth, 0.0, 1.0);
        let new_nutrient = clamp(nutrient + lap_n * 0.25 - growth * 0.85, 0.0, 1.0);
        
        // Iridescent structural color based on crystal orientation & phase
        let irid_r = 0.5 + 0.5 * cos(theta + 0.0);
        let irid_g = 0.5 + 0.5 * cos(theta + 2.094);
        let irid_b = 0.5 + 0.5 * cos(theta + 4.188);
        
        new_state[0] = mix(0.05, irid_r * 0.9 + 0.1, new_crystal);
        new_state[1] = mix(0.08, irid_g * 0.95 + 0.05, new_crystal);
        new_state[2] = mix(0.15, irid_b * 1.0, new_crystal);
        new_state[3] = clamp(new_crystal * 1.3, 0.0, 1.0);
        new_state[4] = new_nutrient;
        new_state[5] = orientation;
        
    } else if (params.model_type == 4u) {
        // ==========================================
        // 5. TURING REACTION-DIFFUSION (Labyrinthine Waves)
        // ==========================================
        let u_val = center_state[0];
        let v_val = center_state[1];
        let lap_u = lap[0];
        let lap_v = lap[1];
        
        // Gray-Scott parameters for labyrinthine morphogenetic patterns
        let du = 0.18;
        let dv = 0.09;
        let f_feed = 0.0545;
        let k_kill = 0.062;
        
        let uvv = u_val * v_val * v_val;
        let du_dt = du * lap_u - uvv + f_feed * (1.0 - u_val);
        let dv_dt = dv * lap_v + uvv - (f_feed + k_kill) * v_val;
        
        let new_u = clamp(u_val + du_dt * 1.5, 0.0, 1.0);
        let new_v = clamp(v_val + dv_dt * 1.5, 0.0, 1.0);
        
        // Color mapping: Activator wave vs Inhibitor valleys
        new_state[0] = clamp(new_v * 2.2, 0.0, 1.0);
        new_state[1] = clamp(new_u * 0.4 + new_v * 0.6, 0.0, 1.0);
        new_state[2] = clamp((1.0 - new_u) * 1.8, 0.0, 1.0);
        new_state[3] = clamp((new_u + new_v) * 0.8, 0.0, 1.0);
    }
    
    // ==============================================
    // INTERACTIVE BRUSH TOOL INTEGRATION
    // ==============================================
    if (params.brush_active == 1u) {
        let cell_pos = vec2<f32>(f32(x), f32(y));
        let brush_pos = vec2<f32>(params.brush_x, params.brush_y);
        let dist = distance(cell_pos, brush_pos);
        
        if (dist <= params.brush_radius) {
            let falloff = 1.0 - smoothstep(params.brush_radius * 0.6, params.brush_radius, dist);
            let strength = params.brush_strength * falloff;
            
            if (params.brush_type == 0u) {
                // Scalpel / Amputation Eraser: Zeroes out cell tissues, carving wounds
                for (var c = 0u; c < 16u; c++) {
                    new_state[c] *= (1.0 - strength);
                }
            } else if (params.brush_type == 1u) {
                // Seed Pen: Inoculates fresh embryonic cells with viable morphogens
                new_state[0] = mix(new_state[0], 0.25, strength);
                new_state[1] = mix(new_state[1], 0.85, strength);
                new_state[2] = mix(new_state[2], 0.45, strength);
                new_state[3] = mix(new_state[3], 1.0, strength);
                for (var c = 4u; c < 16u; c++) {
                    new_state[c] = mix(new_state[c], 1.0, strength);
                }
            } else if (params.brush_type == 2u) {
                // Chemical Mutator Brush: Injects localized chaotic morphogen fluctuations
                for (var c = 4u; c < 16u; c++) {
                    let noise = hash13(vec3<f32>(f32(x) * 1.33 + f32(c), f32(y) * 2.11, params.seed * 4.7)) * 2.0 - 1.0;
                    new_state[c] += noise * strength * 0.6;
                }
            } else if (params.brush_type == 3u) {
                // Nutrient Stimulant: Fuels accelerated mitosis and growth
                new_state[4] = clamp(new_state[4] + strength * 0.5, 0.0, 1.0);
                if (params.model_type == 2u) {
                    new_state[8] = clamp(new_state[8] + strength * 0.8, 0.0, 1.0); // Luminescent flare
                }
            }
        }
    }
    
    // Bounds clamping and final write to Ping-Pong Output Buffer
    new_state[0] = clamp(new_state[0], 0.0, 1.0);
    new_state[1] = clamp(new_state[1], 0.0, 1.0);
    new_state[2] = clamp(new_state[2], 0.0, 1.0);
    new_state[3] = clamp(new_state[3], 0.0, 1.0);
    
    for (var c = 0u; c < 16u; c++) {
        stateOut[current_idx + c] = new_state[c];
    }
}
`;

export const RENDER_SHADER = `
struct RenderParams {
    grid_width: f32,
    grid_height: f32,
    zoom: f32,
    pan_x: f32,
    
    pan_y: f32,
    viewport_width: f32,
    viewport_height: f32,
    display_mode: u32,       // 0=RGBA Specimen, 1=Hidden Morphogen, 2=Alpha Mask, 3=Chemical Gradients
    
    selected_channel: u32,   // 4..15
    brush_active: u32,
    brush_x: f32,
    brush_y: f32,
    
    brush_radius: f32,
    brush_type: u32,
    time: f32,
    padding: f32,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32((vertex_index << 1u) & 2u);
    let y = f32(vertex_index & 2u);
    out.position = vec4<f32>(x * 2.0 - 1.0, y * -2.0 + 1.0, 0.0, 1.0);
    out.uv = vec2<f32>(x, y);
    return out;
}

@group(0) @binding(0) var<storage, read> cell_state: array<f32>;
@group(0) @binding(1) var<uniform> render_params: RenderParams;

// Turbo color map for false-color hidden morphogen channels
fn turbo_colormap(x_in: f32) -> vec3<f32> {
    let x = clamp(x_in, 0.0, 1.0);
    let r = 0.1357 + x * ( 4.6153 + x * (-42.6603 + x * (132.1310 + x * (-152.9423 + x * 59.2863))));
    let g = 0.0914 + x * ( 2.1941 + x * (  4.8429 + x * ( -14.1850 + x * (   4.2772 + x *  2.8295))));
    let b = 0.1067 + x * (12.5925 + x * (-60.1818 + x * (109.0745 + x * ( -88.5064 + x * 26.8195))));
    return clamp(vec3<f32>(r, g, b), vec3<f32>(0.0), vec3<f32>(1.0));
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let screen_px = in.uv * vec2<f32>(render_params.viewport_width, render_params.viewport_height);
    let center = vec2<f32>(render_params.viewport_width, render_params.viewport_height) * 0.5;
    
    // Map screen coordinate through Pan and Zoom to Grid coordinate (gx, gy)
    let gx = (screen_px.x - center.x - render_params.pan_x) / render_params.zoom + render_params.grid_width * 0.5;
    let gy = (screen_px.y - center.y - render_params.pan_y) / render_params.zoom + render_params.grid_height * 0.5;
    
    // Outside simulation dish boundary
    if (gx < 0.0 || gx >= render_params.grid_width || gy < 0.0 || gy >= render_params.grid_height) {
        let grid_pat = step(0.96, fract(gx * 0.1)) + step(0.96, fract(gy * 0.1));
        return vec4<f32>(0.02 + grid_pat * 0.015, 0.03 + grid_pat * 0.02, 0.045 + grid_pat * 0.03, 1.0);
    }
    
    let ix = u32(floor(gx));
    let iy = u32(floor(gy));
    let cell_idx = (iy * u32(render_params.grid_width) + ix) * 16u;
    
    let r = cell_state[cell_idx + 0u];
    let g = cell_state[cell_idx + 1u];
    let b = cell_state[cell_idx + 2u];
    let a = cell_state[cell_idx + 3u];
    
    var color = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    
    if (render_params.display_mode == 0u) {
        // Mode 0: RGBA Synthetic Biology Specimen over Dark Laboratory Substrate
        let petri_bg = vec3<f32>(0.035, 0.042, 0.06);
        
        // Microscopic cellular grid lines when deeply zoomed in
        var cell_border = 0.0;
        if (render_params.zoom > 4.0) {
            let frac_uv = fract(vec2<f32>(gx, gy));
            if (frac_uv.x < 0.06 || frac_uv.y < 0.06) {
                cell_border = 0.06;
            }
        }
        
        let specimen_rgb = vec3<f32>(r, g, b);
        let final_rgb = mix(petri_bg + cell_border, specimen_rgb, clamp(a, 0.0, 1.0));
        color = vec4<f32>(final_rgb, 1.0);
        
    } else if (render_params.display_mode == 1u) {
        // Mode 1: Hidden Morphogen Chemical Channel False-Color Heatmap
        let ch_idx = clamp(render_params.selected_channel, 4u, 15u);
        let ch_val = cell_state[cell_idx + ch_idx];
        let norm_val = clamp(ch_val * 0.5 + 0.5, 0.0, 1.0);
        color = vec4<f32>(turbo_colormap(norm_val), 1.0);
        
    } else if (render_params.display_mode == 2u) {
        // Mode 2: Alive Mask & Biomass Density Channel (Alpha)
        color = vec4<f32>(vec3<f32>(a), 1.0);
        
    } else if (render_params.display_mode == 3u) {
        // Mode 3: Morphogenetic Chemical Gradient Field
        let grad_x = cell_state[cell_idx + 4u] - cell_state[cell_idx + 5u];
        let grad_y = cell_state[cell_idx + 6u] - cell_state[cell_idx + 7u];
        let mag = sqrt(grad_x * grad_x + grad_y * grad_y);
        let angle = atan2(grad_y, grad_x);
        let hue_r = 0.5 + 0.5 * cos(angle);
        let hue_g = 0.5 + 0.5 * cos(angle + 2.094);
        let hue_b = 0.5 + 0.5 * cos(angle + 4.188);
        color = vec4<f32>(vec3<f32>(hue_r, hue_g, hue_b) * clamp(mag * 2.0, 0.1, 1.0), 1.0);
    }
    
    // Draw Brush Reticle Overlay directly in shader for zero-latency feedback
    if (render_params.brush_active == 1u) {
        let b_dist = distance(vec2<f32>(gx, gy), vec2<f32>(render_params.brush_x, render_params.brush_y));
        let ring_width = 1.5 / render_params.zoom;
        if (abs(b_dist - render_params.brush_radius) < ring_width) {
            if (render_params.brush_type == 0u) {
                // Red scalpel ring
                color = mix(color, vec4<f32>(1.0, 0.25, 0.35, 1.0), 0.9);
            } else if (render_params.brush_type == 1u) {
                // Emerald seed pen ring
                color = mix(color, vec4<f32>(0.15, 1.0, 0.5, 1.0), 0.9);
            } else if (render_params.brush_type == 2u) {
                // Cyan chemical mutator ring
                color = mix(color, vec4<f32>(0.2, 0.85, 1.0, 1.0), 0.9);
            } else {
                // Gold nutrient stimulant ring
                color = mix(color, vec4<f32>(1.0, 0.85, 0.2, 1.0), 0.9);
            }
        }
    }
    
    return color;
}
`;
