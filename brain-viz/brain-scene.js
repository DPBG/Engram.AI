/**
 * Main scene — orchestrates the 3D brain visualization.
 * Sets up Three.js renderer, camera, post-processing, and coordinates
 * data flow between DataBridge, BrainRegions, and BrainSynapses.
 */
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';

import { DataBridge, REGIONS } from './data-bridge.js?v=5';
import { BrainRegions } from './brain-regions.js';
import { BrainSynapses } from './brain-synapses.js';

class BrainScene {
    constructor() {
        this.container = document.getElementById('canvas-container');
        this.clock = new THREE.Clock();
        this._lastMetrics = null;

        this._initRenderer();
        this._initScene();
        this._initCamera();
        this._initLights();
        this._initPostProcessing();
        this._initControls();

        // Starfield background for depth
        this._addStarfield();

        // Build brain components
        this.brainRegions = new BrainRegions(this.scene);
        this.brainSynapses = new BrainSynapses(this.scene, this.brainRegions);

        // Data bridge
        this.dataBridge = new DataBridge();
        this.dataBridge.onMetrics((m) => this._onMetrics(m));

        // Parse URL params for configuration
        const params = new URLSearchParams(window.location.search);
        this._liveWsUrl = params.get('ws') || this._detectLiveWsUrl();
        this.embedMode = params.get('embed') === 'true';
        this.showLabels = params.get('labels') !== 'false';

        // Embed mode: hide HUD for clean iframe embedding
        if (this.embedMode) {
            const hud = document.getElementById('hud');
            const status = document.getElementById('connection-status');
            if (hud) hud.style.display = 'none';
            if (status) status.style.display = 'none';
        }

        // Wire up toggle button
        const statusBtn = document.getElementById('connection-status');
        if (statusBtn) {
            statusBtn.addEventListener('click', () => this._toggleMode());
        }

        // Default to live mode — fall back to simulated if connection fails
        if (params.get('mode') === 'sim') {
            this.dataBridge.startSimulated();
        } else {
            this.dataBridge.startLive(this._liveWsUrl);
        }

        // Region labels (HTML overlay)
        this._createRegionLabels();

        // Handle resize
        window.addEventListener('resize', () => this._onResize());

        // Start render loop
        this._animate();
    }

    _detectLiveWsUrl() {
        const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const host = window.location.hostname;
        // demo.engram.ai: nginx on standard port proxies /ws to dashboard
        if (host === 'demo.engram.ai') {
            return `${proto}//${host}/ws`;
        }
        // Direct access: dashboard on port 8080
        return `${proto}//${host}:8080/ws`;
    }

    _toggleMode() {
        if (!this._liveWsUrl) return;
        this.dataBridge.toggle(this._liveWsUrl);
    }

    _initRenderer() {
        this.renderer = new THREE.WebGLRenderer({
            antialias: true,
            alpha: false,
            powerPreference: 'high-performance',
        });
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.renderer.setClearColor(0x0a0e1a, 1);
        this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
        this.renderer.toneMappingExposure = 1.2;
        this.container.appendChild(this.renderer.domElement);
    }

    _initScene() {
        this.scene = new THREE.Scene();
        this.scene.fog = new THREE.FogExp2(0x0a0e1a, 0.014);
    }

    _initCamera() {
        this.camera = new THREE.PerspectiveCamera(
            50,
            window.innerWidth / window.innerHeight,
            0.1,
            100,
        );
        this.camera.position.set(14, 4, 8);
        this.camera.lookAt(0, 0.5, -0.5);
    }

    _initLights() {
        // Subtle ambient
        this.ambientLight = new THREE.AmbientLight(0x1a2a4a, 0.3);
        this.scene.add(this.ambientLight);

        // Key light — soft blue from above (DA shifts this warm)
        this.keyLight = new THREE.PointLight(0x4488cc, 0.5, 30);
        this.keyLight.position.set(5, 10, 5);
        this.scene.add(this.keyLight);

        // Fill light — warm from below
        this.fillLight = new THREE.PointLight(0x442244, 0.3, 25);
        this.fillLight.position.set(-5, -5, -5);
        this.scene.add(this.fillLight);

        // Rim light — bright accent from behind
        this.rimLight = new THREE.PointLight(0x88aaff, 0.4, 20);
        this.rimLight.position.set(-3, 3, -10);
        this.scene.add(this.rimLight);

        // Smoothed neuromodulation for scene effects
        this._smoothNm = { da: 1.2, ach: 1.2, ne: 1.2, serotonin: 1.0 };
    }

    _initPostProcessing() {
        this.composer = new EffectComposer(this.renderer);

        const renderPass = new RenderPass(this.scene, this.camera);
        this.composer.addPass(renderPass);

        // Bloom — tuned for lightning bolt glow
        this.bloomPass = new UnrealBloomPass(
            new THREE.Vector2(window.innerWidth, window.innerHeight),
            0.8,    // strength — stronger for lightning corona
            0.5,    // radius — wider spread for electrical glow
            0.7,    // threshold — lower to catch bolt brightness
        );
        this.composer.addPass(this.bloomPass);
    }

    _initControls() {
        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.rotateSpeed = 0.5;
        this.controls.zoomSpeed = 0.8;
        this.controls.autoRotate = true;
        this.controls.autoRotateSpeed = 0.3;
        this.controls.minDistance = 5;
        this.controls.maxDistance = 25;
        this.controls.target.set(0, 0.5, -0.5);
    }

    _addStarfield() {
        const count = 500;
        const positions = new Float32Array(count * 3);
        for (let i = 0; i < count; i++) {
            positions[i * 3]     = (Math.random() - 0.5) * 80;
            positions[i * 3 + 1] = (Math.random() - 0.5) * 80;
            positions[i * 3 + 2] = (Math.random() - 0.5) * 80;
        }
        const geo = new THREE.BufferGeometry();
        geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        const mat = new THREE.PointsMaterial({
            color: 0x334466,
            size: 0.08,
            transparent: true,
            opacity: 0.4,
        });
        this.scene.add(new THREE.Points(geo, mat));
    }

    _createRegionLabels() {
        this.regionLabels = {};
        const labelContainer = document.createElement('div');
        labelContainer.style.position = 'absolute';
        labelContainer.style.top = '0';
        labelContainer.style.left = '0';
        labelContainer.style.width = '100%';
        labelContainer.style.height = '100%';
        labelContainer.style.pointerEvents = 'none';
        labelContainer.style.overflow = 'visible';
        document.body.appendChild(labelContainer);

        for (const region of REGIONS) {
            const label = document.createElement('div');
            label.className = 'region-label';

            // Label text + firing rate
            const nameSpan = document.createElement('span');
            nameSpan.textContent = region.label;
            const rateSpan = document.createElement('span');
            rateSpan.className = 'rate';
            rateSpan.textContent = '';
            label.appendChild(nameSpan);
            label.appendChild(rateSpan);

            // Tooltip (shown on hover)
            const tooltip = document.createElement('div');
            tooltip.className = 'region-tooltip';
            const ttName = document.createElement('div');
            ttName.className = 'tt-name';
            ttName.textContent = region.label;
            const ttNeurons = document.createElement('div');
            ttNeurons.className = 'tt-neurons';
            ttNeurons.textContent = this._formatNum(region.neurons) + ' neurons';
            const ttDesc = document.createElement('div');
            ttDesc.className = 'tt-desc';
            ttDesc.textContent = region.description || '';
            tooltip.appendChild(ttName);
            tooltip.appendChild(ttNeurons);
            tooltip.appendChild(ttDesc);
            if (region.connections) {
                const ttConn = document.createElement('div');
                ttConn.className = 'tt-connections';
                ttConn.textContent = region.connections;
                tooltip.appendChild(ttConn);
            }
            label.appendChild(tooltip);

            // Pause auto-rotate on hover for easier reading
            label.addEventListener('mouseenter', () => {
                this.controls.autoRotate = false;
            });
            label.addEventListener('mouseleave', () => {
                this.controls.autoRotate = true;
            });

            labelContainer.appendChild(label);
            this.regionLabels[region.id] = { el: label, rateEl: rateSpan, neuronsEl: ttNeurons };
        }
    }

    _updateRegionLabels() {
        if (!this.showLabels) return;
        const halfW = window.innerWidth / 2;
        const halfH = window.innerHeight / 2;

        for (const region of REGIONS) {
            const entry = this.regionLabels[region.id];
            if (!entry) continue;
            const label = entry.el;

            const center = this.brainRegions.getCenter(region.id);
            const projected = center.clone().project(this.camera);

            // Check if behind camera
            if (projected.z > 1) {
                label.style.display = 'none';
                continue;
            }

            const x = projected.x * halfW + halfW;
            const y = -projected.y * halfH + halfH;

            label.style.display = 'block';
            label.style.left = `${x}px`;
            label.style.top = `${y}px`;
            label.style.transform = 'translate(-50%, -50%)';

            // Flip tooltip below if label is in upper portion of screen
            const tooltip = label.querySelector('.region-tooltip');
            if (tooltip) {
                if (y < window.innerHeight * 0.35) {
                    tooltip.style.bottom = 'auto';
                    tooltip.style.top = 'calc(100% + 8px)';
                    tooltip.classList.add('flipped');
                } else {
                    tooltip.style.bottom = 'calc(100% + 8px)';
                    tooltip.style.top = 'auto';
                    tooltip.classList.remove('flipped');
                }
            }

            // Active state when firing
            const data = this.brainRegions.regionData[region.id];
            if (data && data.currentRate > 0.02) {
                label.classList.add('active');
                const pct = (data.currentRate * 100).toFixed(1);
                entry.rateEl.textContent = pct + '%';
                entry.neuronsEl.textContent = this._formatNum(region.neurons) + ' neurons \u00B7 firing ' + pct + '%';
            } else {
                label.classList.remove('active');
                entry.rateEl.textContent = '';
                entry.neuronsEl.textContent = this._formatNum(region.neurons) + ' neurons';
            }
        }
    }

    _onMetrics(metrics) {
        this._lastMetrics = metrics;

        // Update brain regions
        this.brainRegions.setFiringRates(metrics.firingRates);

        // Update synapses
        this.brainSynapses.setFiringRates(metrics.firingRates);
        this.brainSynapses.setNeuromodulation(metrics.neuromodulation);

        // Update HUD
        this._updateHUD(metrics);

        // Update smoothed neuromodulation target
        const nm = metrics.neuromodulation || {};
        if (nm.da !== undefined) this._targetNm = nm;
    }

    /** Scene-wide neuromodulation visual effects */
    _updateNeuromodEffects(dt) {
        if (!this._targetNm) return;
        const lerp = Math.min(1, dt * 2);
        const nm = this._smoothNm;
        const target = this._targetNm;

        nm.da += (( target.da || 1.2) - nm.da) * lerp;
        nm.ne += ((target.ne || 1.2) - nm.ne) * lerp;
        nm.serotonin += ((target.serotonin || 1.0) - nm.serotonin) * lerp;

        // DA → warm shift on key light (high DA = more orange/yellow)
        const daWarm = Math.max(0, (nm.da - 0.8) * 0.3);
        this.keyLight.color.setRGB(
            0.27 + daWarm * 0.4,   // more red when DA high
            0.53 - daWarm * 0.1,   // slightly less green
            0.8  - daWarm * 0.3,   // less blue
        );

        // NE → increase bloom strength slightly (arousal = more glow)
        const neBoost = Math.max(0, (nm.ne - 0.8) * 0.15);
        this.bloomPass.strength = 0.8 + neBoost;

        // 5-HT → cool ambient shift (high serotonin = calmer, bluer)
        const shtCalm = Math.max(0, (nm.serotonin - 0.5) * 0.15);
        this.ambientLight.color.setRGB(
            0.1 - shtCalm * 0.03,
            0.16 + shtCalm * 0.02,
            0.29 + shtCalm * 0.05,
        );
    }

    _updateHUD(m) {
        const el = (id) => document.getElementById(id);

        el('hud-neurons').textContent = this._formatNum(m.totalNeurons);
        el('hud-synapses').textContent = '1.31B';
        el('hud-step').textContent = this._formatNum(m.step);
        el('hud-rate').textContent = m.stepsPerSec.toFixed(2);

        // Phase badge
        const phaseEl = el('hud-phase');
        const phase = (m.neuromodulation && m.neuromodulation.phase) || 'unknown';
        phaseEl.textContent = phase.charAt(0).toUpperCase() + phase.slice(1);
        phaseEl.className = 'hud-phase ' + phase;

        // Neuromodulator bars
        const nm = m.neuromodulation || {};
        const maxNm = 3.0;
        this._setBarHeight('nm-da', (nm.da || 0) / maxNm);
        this._setBarHeight('nm-ach', (nm.ach || 0) / maxNm);
        this._setBarHeight('nm-ne', (nm.ne || 0) / maxNm);
        this._setBarHeight('nm-sht', (nm.serotonin || 0) / maxNm);
    }

    _setBarHeight(id, pct) {
        const el = document.getElementById(id);
        if (el) el.style.height = `${Math.min(100, Math.max(0, pct * 100))}%`;
    }

    _formatNum(n) {
        if (n >= 1000000) return (n / 1000000).toFixed(2) + 'M';
        if (n >= 1000) return (n / 1000).toFixed(1) + 'K';
        return String(n);
    }

    _onResize() {
        const w = window.innerWidth;
        const h = window.innerHeight;
        this.camera.aspect = w / h;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(w, h);
        this.composer.setSize(w, h);
        this.bloomPass.resolution.set(w, h);
    }

    _animate() {
        requestAnimationFrame(() => this._animate());
        const dt = Math.min(this.clock.getDelta(), 0.05); // Cap dt to avoid jumps

        // Update components
        this.controls.update();
        this.brainRegions.update(dt);
        this.brainSynapses.update(dt);
        this._updateNeuromodEffects(dt);

        // Update region labels
        this._updateRegionLabels();

        // Render with post-processing
        this.composer.render();
    }
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => new BrainScene());
} else {
    new BrainScene();
}
