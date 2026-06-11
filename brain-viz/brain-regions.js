/**
 * Brain region particle clusters with anatomically-shaped distributions
 * and a transparent brain hull mesh for recognizable silhouette.
 */
import * as THREE from 'three';
import { REGIONS, REGION_POSITIONS, REGION_SHAPES } from './data-bridge.js';

// Particle counts per region (scaled down from real neuron count)
function particleCount(neurons) {
    // Log scale: 1800 → ~80, 500000 → ~1200
    return Math.floor(60 + Math.log2(neurons + 1) * 60);
}

// Gaussian random with Box-Muller
function gaussRandom() {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

export class BrainRegions {
    constructor(scene) {
        this.scene = scene;
        this.regionMeshes = {};   // id → InstancedMesh
        this.regionData = {};     // id → { count, baseColors, currentRate }
        this.regionCenters = {};  // id → THREE.Vector3
        this.time = 0;
        this._basePositions = {}; // id → Float32Array (x,y,z per instance)
        this._noisePhases = {};   // id → Float32Array (3 phases per instance)

        this._build();
        this._buildBrainHull();
    }

    _build() {
        const geo = new THREE.SphereGeometry(0.04, 6, 4);

        for (const region of REGIONS) {
            const pos = REGION_POSITIONS[region.id];
            const center = new THREE.Vector3(pos[0], pos[1], pos[2]);
            this.regionCenters[region.id] = center;

            const count = particleCount(region.neurons);
            const mat = new THREE.MeshBasicMaterial({
                color: new THREE.Color(region.color[0] * 0.6, region.color[1] * 0.6, region.color[2] * 0.6),
                transparent: true,
                opacity: 0.4,
            });

            const mesh = new THREE.InstancedMesh(geo, mat, count);
            mesh.frustumCulled = false;

            const shapeConfig = REGION_SHAPES[region.id] || { spread: [1, 1, 1], shape: 'ellipsoid' };
            const [sx, sy, sz] = shapeConfig.spread;

            const dummy = new THREE.Matrix4();
            const basePositions = new Float32Array(count * 3);
            const noisePhases = new Float32Array(count * 3);

            for (let i = 0; i < count; i++) {
                let lx, ly, lz;

                switch (shapeConfig.shape) {
                    case 'shell': {
                        // Particles on a spherical shell surface — cortical appearance
                        const nx = gaussRandom(), ny = gaussRandom(), nz = gaussRandom();
                        const len = Math.sqrt(nx * nx + ny * ny + nz * nz) || 1;
                        const r = shapeConfig.shellRadius + gaussRandom() * shapeConfig.shellThickness;
                        lx = (nx / len) * r * sx * 0.35;
                        ly = (ny / len) * r * sy * 0.35;
                        lz = (nz / len) * r * sz * 0.35;
                        break;
                    }
                    case 'cylinder': {
                        // Tall narrow cylinder — brainstem shape
                        const angle = Math.random() * Math.PI * 2;
                        const r = Math.sqrt(Math.random()) * sx;
                        lx = Math.cos(angle) * r;
                        ly = gaussRandom() * sy;
                        lz = Math.sin(angle) * sz;
                        break;
                    }
                    default: { // 'ellipsoid'
                        lx = gaussRandom() * sx;
                        ly = gaussRandom() * sy;
                        lz = gaussRandom() * sz;
                        break;
                    }
                }

                const x = center.x + lx;
                const y = center.y + ly;
                const z = center.z + lz;

                basePositions[i * 3] = x;
                basePositions[i * 3 + 1] = y;
                basePositions[i * 3 + 2] = z;

                dummy.setPosition(x, y, z);
                mesh.setMatrixAt(i, dummy);

                // Random phase offsets for breathing
                noisePhases[i * 3] = Math.random() * Math.PI * 2;
                noisePhases[i * 3 + 1] = Math.random() * Math.PI * 2;
                noisePhases[i * 3 + 2] = Math.random() * Math.PI * 2;
            }

            mesh.instanceMatrix.needsUpdate = true;
            this.scene.add(mesh);

            this.regionMeshes[region.id] = mesh;
            this.regionData[region.id] = {
                count,
                center,
                baseColor: new THREE.Color(region.color[0], region.color[1], region.color[2]),
                currentRate: 0,
                targetRate: 0,
                pulseIntensity: 0,  // flash-on-activate effect
                prevTargetRate: 0,  // track rate changes for pulse trigger
            };
            this._basePositions[region.id] = basePositions;
            this._noisePhases[region.id] = noisePhases;
        }
    }

    /** Procedural brain hull — deformed sphere for recognizable silhouette */
    _buildBrainHull() {
        const geo = new THREE.SphereGeometry(5.5, 32, 24);
        const pos = geo.attributes.position;

        for (let i = 0; i < pos.count; i++) {
            let x = pos.getX(i);
            let y = pos.getY(i);
            let z = pos.getZ(i);

            // Flatten bottom (brainstem exit) — pinch below y=-2
            if (y < -2.0) {
                const squeeze = 1.0 - Math.max(0, (-y - 2.0) * 0.35);
                x *= Math.max(0.12, squeeze);
                z *= Math.max(0.12, squeeze);
            }

            // Extend front (frontal lobe bulge)
            if (z > 0) {
                z *= 1.3;
                if (y > 0) y *= 1.0 + z * 0.03;
            }

            // Posterior indent above cerebellum (tentorium separation)
            if (z < -2.0 && y < 0 && y > -2.5) {
                const indent = Math.max(0, (-z - 2.0) * 0.12);
                x *= 1.0 - indent * 0.25;
            }

            // Slight bilateral compression (brain is narrower left-right than front-back)
            x *= 0.6;

            // Cerebellum bulge at back-bottom
            if (z < -2.5 && y < -0.5 && y > -3.0) {
                const bulge = Math.sin((y + 0.5) / 2.5 * Math.PI) * 0.5;
                z -= bulge;
            }

            // Shift center to match region layout (slightly posterior-upper)
            y += 0.5;
            z -= 0.5;

            pos.setXYZ(i, x, y, z);
        }

        geo.computeVertexNormals();

        // Subtle solid fill with procedural sulci pattern
        const sulciMat = new THREE.ShaderMaterial({
            transparent: true,
            depthWrite: false,
            side: THREE.DoubleSide,
            blending: THREE.NormalBlending,
            uniforms: {
                baseColor: { value: new THREE.Color(0x2a3a5a) },
            },
            vertexShader: `
                varying vec3 vPos;
                void main() {
                    vPos = position;
                    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                }
            `,
            fragmentShader: `
                uniform vec3 baseColor;
                varying vec3 vPos;
                void main() {
                    // Procedural sulci/gyri pattern — wavy lines suggesting cortical folds
                    float sulci = sin(vPos.x * 12.0 + sin(vPos.y * 4.0) * 2.0)
                                * sin(vPos.y * 8.0 + sin(vPos.z * 5.0) * 1.5) * 0.5 + 0.5;
                    float alpha = mix(0.02, 0.07, sulci);
                    gl_FragColor = vec4(baseColor, alpha);
                }
            `,
        });
        const solidMesh = new THREE.Mesh(geo, sulciMat);
        this.scene.add(solidMesh);

        // Wireframe overlay for structural hint
        const wireMat = new THREE.MeshBasicMaterial({
            color: 0x3a5a8a,
            transparent: true,
            opacity: 0.07,
            wireframe: true,
            depthWrite: false,
        });
        const wireMesh = new THREE.Mesh(geo.clone(), wireMat);
        this.scene.add(wireMesh);

        this._hullMeshes = [solidMesh, wireMesh];
    }

    /** Update firing rates for all regions */
    setFiringRates(firingRates) {
        for (const region of REGIONS) {
            const rate = firingRates[region.id] || 0;
            if (this.regionData[region.id]) {
                this.regionData[region.id].targetRate = rate;
            }
        }
    }

    /** Animate particles — call every frame */
    update(dt) {
        this.time += dt;
        const dummy = new THREE.Matrix4();

        for (const region of REGIONS) {
            const data = this.regionData[region.id];
            const mesh = this.regionMeshes[region.id];
            if (!data || !mesh) continue;

            // Smooth interpolation of firing rate
            data.currentRate += (data.targetRate - data.currentRate) * Math.min(1, dt * 3);

            // Detect activation: target rate jumped up → trigger pulse
            if (data.targetRate > data.prevTargetRate + 0.005) {
                data.pulseIntensity = 1.0;
            }
            data.prevTargetRate = data.targetRate;

            // Decay pulse
            data.pulseIntensity *= Math.max(0, 1.0 - dt * 3.0);
            if (data.pulseIntensity < 0.01) data.pulseIntensity = 0;

            // Map firing rate to visual intensity using sqrt curve for wider spread
            // 0.01 → 0.22, 0.03 → 0.39, 0.05 → 0.50, 0.08 → 0.63, 0.12 → 0.77, 0.20 → 1.0
            const baseIntensity = Math.min(1.0, Math.sqrt(data.currentRate * 5.0));

            // Add pulse flash on top (additive, capped)
            const intensity = Math.min(1.0, baseIntensity + data.pulseIntensity * 0.4);

            // Update material opacity and brightness
            mesh.material.opacity = 0.15 + intensity * 0.7;
            const col = mesh.material.color;
            const bright = 0.25 + intensity * 0.55; // max 0.8
            col.r = data.baseColor.r * bright + data.pulseIntensity * 0.15;
            col.g = data.baseColor.g * bright + data.pulseIntensity * 0.1;
            col.b = data.baseColor.b * bright + data.pulseIntensity * 0.15;

            // Breathing animation: oscillate around base position (no drift)
            const basePos = this._basePositions[region.id];
            const phases = this._noisePhases[region.id];
            const breathAmp = 0.02 + intensity * 0.05;
            const breathSpeed = 0.5 + intensity * 1.5;
            const t = this.time;

            for (let i = 0; i < data.count; i++) {
                const bx = basePos[i * 3];
                const by = basePos[i * 3 + 1];
                const bz = basePos[i * 3 + 2];
                const px = phases[i * 3];
                const py = phases[i * 3 + 1];
                const pz = phases[i * 3 + 2];

                // Sine-based oscillation from fixed base position
                const dx = Math.sin(t * breathSpeed + px) * breathAmp;
                const dy = Math.sin(t * breathSpeed * 0.8 + py) * breathAmp * 0.7;
                const dz = Math.sin(t * breathSpeed * 0.6 + pz) * breathAmp * 0.8;

                dummy.setPosition(bx + dx, by + dy, bz + dz);
                mesh.setMatrixAt(i, dummy);
            }

            mesh.instanceMatrix.needsUpdate = true;
        }
    }

    /** Get center position of a region */
    getCenter(regionId) {
        return this.regionCenters[regionId] || new THREE.Vector3();
    }

    dispose() {
        for (const mesh of Object.values(this.regionMeshes)) {
            mesh.geometry.dispose();
            mesh.material.dispose();
            this.scene.remove(mesh);
        }
        if (this._hullMeshes) {
            for (const mesh of this._hullMeshes) {
                mesh.geometry.dispose();
                mesh.material.dispose();
                this.scene.remove(mesh);
            }
        }
    }
}
