/**
 * Data bridge for brain visualization.
 * Connects to dashboard WebSocket for live data, or generates simulated data.
 */

export const REGIONS = [
    {
        id: 'brainstem', label: 'Brainstem', neurons: 1800, color: [1.0, 0.3, 0.2],
        description: 'Manages basic survival drives like energy, temperature, and fatigue. Converts raw sensor signals into neural spikes. Always active — the brain\'s heartbeat.',
        connections: 'Feeds sensory cortex and motor cortex',
    },
    {
        id: 'reflex_arc', label: 'Reflex Arc', neurons: 2000, color: [1.0, 0.6, 0.2],
        description: 'Ultra-fast sensory-to-motor pathway that bypasses higher cognition. Handles immediate danger responses in under 10ms — like pulling away from heat.',
        connections: 'Receives from sensory cortex, drives motor cortex',
    },
    {
        id: 'sensory_cortex', label: 'Sensory Cortex', neurons: 200000, color: [0.13, 0.83, 0.93],
        description: 'Processes all incoming sensory data — vision, audio, touch, proprioception. Each modality occupies a dedicated sub-region. The brain\'s primary input layer.',
        connections: 'Sends to association, motor, cerebellum, features',
    },
    {
        id: 'motor_cortex', label: 'Motor Cortex', neurons: 100000, color: [0.22, 0.85, 0.48],
        description: 'Generates movement commands for locomotion, manipulation, head control, and speech. Contains 6 specialized sub-ranges including a cognitive action channel.',
        connections: 'Receives from all regions, outputs motor commands',
    },
    {
        id: 'cerebellum', label: 'Cerebellum', neurons: 50000, color: [0.95, 0.85, 0.2],
        description: 'Learns precise timing and coordination through error correction. Smooths motor output and builds internal models of body dynamics.',
        connections: 'Receives from sensory, refines motor output',
    },
    {
        id: 'association_cortex', label: 'Association', neurons: 500000, color: [0.37, 0.64, 0.96],
        description: 'The brain\'s largest region — binds different sensory modalities together. "Seeing a face while hearing a voice" creates cross-modal associations via STDP learning.',
        connections: 'Hub connecting all other regions',
    },
    {
        id: 'predictive_layer', label: 'Predictive', neurons: 100000, color: [0.65, 0.45, 0.96],
        description: 'Continuously predicts what sensory input comes next. When prediction error is high, the brain pays attention and learns faster. Drives curiosity and surprise.',
        connections: 'Bidirectional with association and concepts',
    },
    {
        id: 'working_memory', label: 'Working Memory', neurons: 20000, color: [0.96, 0.45, 0.71],
        description: 'Holds recent context for short-term reasoning. Sustained firing patterns maintain information across multiple time steps — like keeping a thought in mind.',
        connections: 'Receives from association and concepts, drives motor',
    },
    {
        id: 'feature_layer', label: 'Feature Layer', neurons: 20000, color: [0.13, 0.78, 0.75],
        description: 'Extracts intermediate features from raw sensory input — edges, textures, phonemes. Learns hierarchical representations automatically through STDP.',
        connections: 'Sits between sensory cortex and association',
    },
    {
        id: 'concept_layer', label: 'Concept Layer', neurons: 5000, color: [0.96, 0.73, 0.15],
        description: 'Forms abstract concepts using winner-take-all competition. Sparse representations where only a few neurons fire for each concept — like how "dog" is a single idea.',
        connections: 'Receives from association and predictive layers',
    },
    {
        id: 'meta_controller', label: 'Meta Control', neurons: 3000, color: [0.9, 0.9, 0.95],
        description: 'Top-level executive control — modulates attention, gates learning, and coordinates global brain state. The closest analog to conscious decision-making.',
        connections: 'Modulates association and motor cortex',
    },
    {
        id: 'pattern_separator', label: 'Pattern Sep.', neurons: 50000, color: [0.6, 0.9, 0.5],
        description: 'Dentate gyrus analog — sparse expansion via k-WTA. Separates similar inputs into distinct representations to prevent interference during memory formation.',
        connections: 'Receives from sensory cortex, feeds concept layer',
    },
    {
        id: 'global_workspace', label: 'Workspace', neurons: 5000, color: [1.0, 1.0, 0.8],
        description: 'Global Neuronal Workspace — ignition-based broadcast for conscious access. When activity exceeds threshold, signals are broadcast to all cortical regions simultaneously.',
        connections: 'Receives from concept and meta, broadcasts to all',
    },
];

// 3D positions — anatomically-inspired sagittal brain layout
// X = left-right (bilateral, kept near 0 for sagittal view)
// Y = up-down (dorsal-ventral)
// Z = front-back (anterior +Z, posterior -Z)
export const REGION_POSITIONS = {
    brainstem:          [ 0.0, -4.0, -1.5],   // narrow stalk at bottom, slightly posterior
    reflex_arc:         [ 0.0, -3.5, -0.5],   // within brainstem area, slightly anterior
    sensory_cortex:     [ 0.0,  2.0, -4.0],   // posterior-upper (parietal/occipital)
    motor_cortex:       [ 0.0,  2.5,  1.5],   // anterior precentral gyrus
    cerebellum:         [ 0.0, -2.5, -4.5],   // posterior-inferior, behind brainstem
    association_cortex: [ 0.0,  1.5, -1.0],   // large central mass (parietal/temporal)
    predictive_layer:   [ 0.0,  3.0,  3.5],   // prefrontal (anterior-superior)
    working_memory:     [ 0.0,  3.5,  2.0],   // dorsolateral prefrontal
    feature_layer:      [ 0.0,  1.0, -3.0],   // secondary visual/auditory
    concept_layer:      [ 0.0, -0.5, -0.5],   // temporal lobe
    meta_controller:    [ 0.0,  4.5,  1.0],   // dorsomedial prefrontal (top-front)
    pattern_separator:  [ 0.0, -1.5, -2.0],   // hippocampal (medial temporal, near concept_layer)
    global_workspace:   [ 0.0,  3.5,  3.0],   // anterior prefrontal (near working memory)
};

// Per-region shape configuration for anatomically-shaped particle clouds
// shape: 'ellipsoid' (default), 'shell' (cortical surface), 'cylinder' (brainstem)
export const REGION_SHAPES = {
    brainstem:          { spread: [0.4, 0.8, 0.4], shape: 'cylinder' },
    reflex_arc:         { spread: [0.5, 0.5, 0.5], shape: 'ellipsoid' },
    sensory_cortex:     { spread: [1.5, 1.8, 1.0], shape: 'shell', shellRadius: 3.0, shellThickness: 0.3 },
    motor_cortex:       { spread: [1.2, 1.8, 0.6], shape: 'shell', shellRadius: 3.5, shellThickness: 0.3 },
    cerebellum:         { spread: [1.5, 0.8, 1.2], shape: 'ellipsoid' },
    association_cortex: { spread: [1.8, 1.5, 2.5], shape: 'ellipsoid' },
    predictive_layer:   { spread: [1.2, 1.0, 0.8], shape: 'shell', shellRadius: 2.5, shellThickness: 0.3 },
    working_memory:     { spread: [0.8, 0.6, 0.7], shape: 'ellipsoid' },
    feature_layer:      { spread: [1.0, 0.8, 0.5], shape: 'ellipsoid' },
    concept_layer:      { spread: [0.8, 0.7, 0.8], shape: 'ellipsoid' },
    meta_controller:    { spread: [0.7, 0.4, 0.6], shape: 'ellipsoid' },
    pattern_separator:  { spread: [1.0, 0.6, 0.8], shape: 'ellipsoid' },
    global_workspace:   { spread: [0.6, 0.5, 0.6], shape: 'ellipsoid' },
};

// Synapse pathway definitions (source -> target) for 3D visualization
export const SYNAPSE_PATHWAYS = [
    { id: 'brainstem_sensory',       src: 'brainstem',         dst: 'sensory_cortex' },
    { id: 'sensory_reflex',          src: 'sensory_cortex',    dst: 'reflex_arc' },
    { id: 'sensory_association',     src: 'sensory_cortex',    dst: 'association_cortex' },
    { id: 'sensory_motor',           src: 'sensory_cortex',    dst: 'motor_cortex' },
    { id: 'sensory_cerebellum',      src: 'sensory_cortex',    dst: 'cerebellum' },
    { id: 'sensory_feature',         src: 'sensory_cortex',    dst: 'feature_layer' },
    { id: 'association_lateral',     src: 'association_cortex', dst: 'association_cortex' },
    { id: 'association_predictive',  src: 'association_cortex', dst: 'predictive_layer' },
    { id: 'association_working',     src: 'association_cortex', dst: 'working_memory' },
    { id: 'association_meta',        src: 'association_cortex', dst: 'meta_controller' },
    { id: 'association_concept',     src: 'association_cortex', dst: 'concept_layer' },
    { id: 'predictive_association',  src: 'predictive_layer',  dst: 'association_cortex' },
    { id: 'predictive_recurrent',    src: 'predictive_layer',  dst: 'predictive_layer' },
    { id: 'predictive_concept',      src: 'predictive_layer',  dst: 'concept_layer' },
    { id: 'reflex_motor',            src: 'reflex_arc',        dst: 'motor_cortex' },
    { id: 'brainstem_motor',         src: 'brainstem',         dst: 'motor_cortex' },
    { id: 'cerebellum_motor',        src: 'cerebellum',        dst: 'motor_cortex' },
    { id: 'working_motor',           src: 'working_memory',    dst: 'motor_cortex' },
    { id: 'feature_association',     src: 'feature_layer',     dst: 'association_cortex' },
    { id: 'concept_lateral',         src: 'concept_layer',     dst: 'concept_layer' },
    { id: 'concept_predictive',      src: 'concept_layer',     dst: 'predictive_layer' },
    { id: 'concept_working',         src: 'concept_layer',     dst: 'working_memory' },
    { id: 'meta_association',        src: 'meta_controller',   dst: 'association_cortex' },
    { id: 'meta_motor',             src: 'meta_controller',   dst: 'motor_cortex' },
    // Pattern separator pathways
    { id: 'sensory_pattern_sep',     src: 'sensory_cortex',    dst: 'pattern_separator' },
    { id: 'pattern_sep_concept',     src: 'pattern_separator', dst: 'concept_layer' },
    // Global workspace pathways
    { id: 'concept_workspace',       src: 'concept_layer',     dst: 'global_workspace' },
    { id: 'workspace_association',   src: 'global_workspace',  dst: 'association_cortex' },
    { id: 'workspace_motor',         src: 'global_workspace',  dst: 'motor_cortex' },
    { id: 'meta_workspace',          src: 'meta_controller',   dst: 'global_workspace' },
];

export class DataBridge {
    constructor() {
        this.mode = 'live'; // 'live' or 'simulated'
        this.ws = null;
        this.listeners = [];
        this.lastMetrics = null;
        this.lastStep = 0;
        this.lastStepTime = 0;
        this.stepsPerSec = 0;
        this._simTime = 0;
        this._simRunning = false;
        this._simPhaseOffsets = REGIONS.map(() => Math.random() * Math.PI * 2);
        this._wsUrl = null;
        this._demoOverride = null; // External firing rate override from parent frame

        // Listen for postMessage from parent (demo pages)
        window.addEventListener('message', (e) => this._handlePostMessage(e));
    }

    /** Register a callback for metrics updates: fn(metrics) */
    onMetrics(fn) {
        this.listeners.push(fn);
    }

    /** Start simulated data mode */
    startSimulated() {
        this._disconnectWs();
        this.mode = 'simulated';
        this._updateConnectionStatus('simulated');
        this._startSimLoop();
    }

    /** Switch to live mode — connect to a WebSocket */
    startLive(wsUrl) {
        this._wsUrl = wsUrl;
        this._stopSimLoop();
        this._connectWs(wsUrl);
    }

    /** Toggle between live and simulated */
    toggle(wsUrl) {
        if (this.mode === 'simulated') {
            this.startLive(wsUrl);
        } else {
            this.startSimulated();
        }
    }

    _connectWs(wsUrl) {
        this._disconnectWs();
        this.mode = 'connecting';
        this._updateConnectionStatus('connecting');
        this._reconnectUrl = wsUrl;
        try {
            this.ws = new WebSocket(wsUrl);
            this.ws.onopen = () => {
                console.log('[DataBridge] Connected to', wsUrl);
                // If we don't get a neuro_update within 8s, start sim alongside
                // but keep trying to reconnect
                this._liveTimeout = setTimeout(() => {
                    if (this.mode === 'connecting') {
                        console.log('[DataBridge] No brain data yet, showing simulated while waiting');
                        this.mode = 'simulated';
                        this._updateConnectionStatus('simulated');
                        this._startSimLoop();
                    }
                }, 8000);
            };
            this.ws.onmessage = (event) => {
                if (this._liveTimeout) { clearTimeout(this._liveTimeout); this._liveTimeout = null; }
                try {
                    const msg = JSON.parse(event.data);
                    this._handleMessage(msg);
                } catch (e) { console.warn('[DataBridge] Parse error:', e.message); }
            };
            this.ws.onclose = () => {
                console.log('[DataBridge] Disconnected, reconnecting in 3s...');
                if (this._liveTimeout) { clearTimeout(this._liveTimeout); this._liveTimeout = null; }
                this._updateConnectionStatus('disconnected');
                // Always try to reconnect — never give up on live
                setTimeout(() => {
                    if (this._reconnectUrl) {
                        this._connectWs(this._reconnectUrl);
                    }
                }, 3000);
                // Show simulated data while reconnecting
                if (!this._simRunning) {
                    this.mode = 'simulated';
                    this._startSimLoop();
                }
            };
            this.ws.onerror = () => {
                console.log('[DataBridge] Connection error, will retry');
                // onclose will fire after onerror, which handles reconnection
            };
        } catch (e) {
            this.mode = 'simulated';
            this._updateConnectionStatus('simulated');
            this._startSimLoop();
            // Retry after delay
            setTimeout(() => {
                if (this._reconnectUrl) this._connectWs(this._reconnectUrl);
            }, 5000);
        }
    }

    _disconnectWs() {
        if (this.ws) {
            try { this.ws.close(); } catch (e) { /* ok */ }
            this.ws = null;
        }
    }

    _handleMessage(msg) {
        if (msg.type === 'neuro_update' && msg.data) {
            if (this.mode !== 'live') {
                this.mode = 'live';
                this._updateConnectionStatus('live');
                this._stopSimLoop();
                console.log('[DataBridge] Receiving live brain data');
            }
            const data = msg.data;
            const now = Date.now();
            if (this.lastStep > 0 && data.step_count > this.lastStep) {
                const dt = (now - this.lastStepTime) / 1000;
                if (dt > 0) this.stepsPerSec = (data.step_count - this.lastStep) / dt;
            }
            this.lastStep = data.step_count || 0;
            this.lastStepTime = now;

            const firingRates = data.firing_rates || {};
            let neuromodulation = data.neuromodulation || {};

            // Apply demo override if active (from parent frame postMessage)
            // This ensures cascade visuals work even when receiving live data
            if (this._demoOverride) {
                const age = Date.now() - this._demoOverride.timestamp;
                if (age > 5000) {
                    this._demoOverride = null;
                } else {
                    const overrideFR = this._demoOverride.firingRates;
                    for (const key of Object.keys(overrideFR)) {
                        if (overrideFR[key] > 0) {
                            firingRates[key] = Math.max(firingRates[key] || 0, overrideFR[key]);
                        }
                    }
                    const overrideNM = this._demoOverride.neuromodulation;
                    if (overrideNM) {
                        neuromodulation = { ...neuromodulation };
                        for (const key of Object.keys(overrideNM)) {
                            if (overrideNM[key] !== undefined) neuromodulation[key] = overrideNM[key];
                        }
                    }
                }
            }

            const metrics = {
                step: data.step_count || 0,
                totalNeurons: data.total_neurons || 1056800,
                stepsPerSec: this.stepsPerSec,
                firingRates,
                neuromodulation,
                drives: data.drives || {},
                synapseStats: data.synapse_stats || {},
                stdpDeltas: data.stdp_deltas || {},
                convergence: data.convergence || {},
            };
            this.lastMetrics = metrics;
            this._notify(metrics);
        } else if (msg.type === 'init' && msg.data) {
            const neuro = msg.data.neuromorphic;
            if (neuro && neuro.step_count) {
                this._handleMessage({ type: 'neuro_update', data: neuro });
            }
        }
    }

    _notify(metrics) {
        for (const fn of this.listeners) {
            try { fn(metrics); } catch (e) { console.error('[DataBridge] Listener error:', e); }
        }
    }

    _updateConnectionStatus(status) {
        const el = document.getElementById('connection-status');
        if (!el) return;
        el.className = 'status-' + status;
        const labels = {
            live: 'LIVE',
            simulated: 'SIMULATED',
            connecting: 'CONNECTING...',
            disconnected: 'DISCONNECTED',
        };
        // Only update the text inside the span, keep the button structure
        const span = el.querySelector('.status-text');
        if (span) {
            span.textContent = labels[status] || status.toUpperCase();
        } else {
            el.textContent = labels[status] || status.toUpperCase();
        }
    }

    /** Handle postMessage from parent frame (demo reaction pages) */
    _handlePostMessage(event) {
        const msg = event.data;
        if (!msg || typeof msg !== 'object') return;

        if (msg.type === 'demo_reaction') {
            this._demoOverride = {
                firingRates: msg.firingRates || {},
                neuromodulation: msg.neuromodulation || {},
                timestamp: Date.now(),
            };
        } else if (msg.type === 'demo_clear') {
            this._demoOverride = null;
        }
    }

    _startSimLoop() {
        if (this._simRunning) return;
        this._simRunning = true;
        console.log('[DataBridge] Running in simulated mode');
        const step = () => {
            if (!this._simRunning) return;
            this._simTime += 0.02;
            const metrics = this._generateSimulatedMetrics();
            this._notify(metrics);
            requestAnimationFrame(step);
        };
        requestAnimationFrame(step);
    }

    _stopSimLoop() {
        this._simRunning = false;
    }

    _generateSimulatedMetrics() {
        const t = this._simTime;
        const firingRates = {};
        REGIONS.forEach((r, i) => {
            const phase = this._simPhaseOffsets[i];
            const base = 0.02 + 0.015 * Math.sin(t * 0.3 + phase);
            const burst = Math.max(0, Math.sin(t * 0.7 + phase * 2) * 0.03);
            const noise = (Math.random() - 0.5) * 0.005;
            firingRates[r.id] = Math.max(0.001, base + burst + noise);
        });

        const sensoryBurst = Math.max(0, Math.sin(t * 0.5) * 0.04);
        firingRates.sensory_cortex += sensoryBurst;
        firingRates.association_cortex += sensoryBurst * 0.6 * Math.max(0, Math.sin(t * 0.5 - 0.5));
        firingRates.motor_cortex += sensoryBurst * 0.3 * Math.max(0, Math.sin(t * 0.5 - 1.0));

        let neuromodulation = {
            phase: 'juvenile',
            da: 1.2 + 0.1 * Math.sin(t * 0.15),
            ach: 1.2 + 0.08 * Math.sin(t * 0.12 + 1),
            ne: 1.2 + 0.15 * Math.sin(t * 0.2 + 2),
            serotonin: 1.0 + 0.05 * Math.sin(t * 0.1 + 3),
            plasticity_multiplier: 0.48,
        };

        // Apply demo override if active (from parent frame postMessage)
        if (this._demoOverride) {
            const age = Date.now() - this._demoOverride.timestamp;
            if (age > 5000) {
                this._demoOverride = null; // Expire after 5 seconds
            } else {
                const overrideFR = this._demoOverride.firingRates;
                for (const key of Object.keys(overrideFR)) {
                    if (overrideFR[key] > 0) {
                        firingRates[key] = Math.max(firingRates[key], overrideFR[key]);
                    }
                }
                const overrideNM = this._demoOverride.neuromodulation;
                if (overrideNM) {
                    for (const key of Object.keys(overrideNM)) {
                        if (overrideNM[key] !== undefined) neuromodulation[key] = overrideNM[key];
                    }
                }
            }
        }

        const step = Math.floor(398000 + t * 0.82);
        return {
            step,
            totalNeurons: 1056800,
            stepsPerSec: 0.82,
            firingRates,
            neuromodulation,
            drives: {
                energy: 0.85 + 0.05 * Math.sin(t * 0.05),
                damage: 0.05,
                temperature: 0.5,
                fatigue: 0.15 + 0.05 * Math.sin(t * 0.08),
            },
            synapseStats: {},
            stdpDeltas: {},
            convergence: { is_converged: false, mean_delta: 0.001 },
        };
    }
}
