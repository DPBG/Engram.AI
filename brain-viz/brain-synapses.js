/**
 * Lightning bolt synapse system.
 * Renders jagged electrical arcs between brain regions with branching,
 * flickering, and bloom-fed glow. Replaces the old smooth trail system.
 */
import * as THREE from 'three';
import { SYNAPSE_PATHWAYS } from './data-bridge.js';

const MAX_BOLT_SEGMENTS = 2000;  // Total line segments across all active bolts
const SPAWN_INTERVAL = 0.05;     // Min seconds between spawns per pathway

// ── Lightning bolt primitive ────────────────────────────────────────

class LightningBolt {
    constructor(curve, color, intensity) {
        this.curve = curve;
        this.color = color;           // THREE.Color
        this.intensity = intensity;   // 0-1
        this.age = 0;
        this.maxAge = 0.10 + Math.random() * 0.20; // 100-300ms flash
        this.segments = 14 + Math.floor(intensity * 18); // 14-32 segments
        this.jitter = 0.25 + intensity * 0.45;  // perpendicular displacement (wider zigzag)

        this.points = this._generateBoltPoints();
        this.branchPoints = this._generateBranches();
    }

    _generateBoltPoints() {
        const pts = [];
        const up = new THREE.Vector3(0, 1, 0);
        const alt = new THREE.Vector3(1, 0, 0);

        for (let i = 0; i <= this.segments; i++) {
            const t = i / this.segments;
            const basePoint = this.curve.getPointAt(t);

            // Endpoints anchor exactly to region centers
            if (i === 0 || i === this.segments) {
                pts.push(basePoint);
                continue;
            }

            // Perpendicular displacement directions from curve tangent
            const tangent = this.curve.getTangentAt(t).normalize();
            let perp1 = new THREE.Vector3().crossVectors(tangent, up);
            if (perp1.lengthSq() < 0.01) perp1.crossVectors(tangent, alt);
            perp1.normalize();
            const perp2 = new THREE.Vector3().crossVectors(tangent, perp1).normalize();

            // Jagged displacement — zigzag pattern
            const d1 = (Math.random() - 0.5) * 2 * this.jitter;
            const d2 = (Math.random() - 0.5) * 2 * this.jitter;

            basePoint.addScaledVector(perp1, d1);
            basePoint.addScaledVector(perp2, d2);
            pts.push(basePoint);
        }
        return pts;
    }

    _generateBranches() {
        const numBranches = Math.floor(this.intensity * 3 * Math.random());
        const branches = [];

        for (let b = 0; b < numBranches; b++) {
            const forkIdx = 2 + Math.floor(Math.random() * Math.max(1, this.points.length - 4));
            if (forkIdx >= this.points.length) continue;

            const forkPoint = this.points[forkIdx];
            const branchLen = 2 + Math.floor(Math.random() * 4);
            const branchPts = [forkPoint.clone()];

            // Branch direction: roughly away from main path
            const prevPt = this.points[Math.max(0, forkIdx - 1)];
            const nextPt = this.points[Math.min(this.points.length - 1, forkIdx + 1)];
            const mainDir = new THREE.Vector3().subVectors(nextPt, prevPt).normalize();
            const branchDir = new THREE.Vector3(
                (Math.random() - 0.5),
                (Math.random() - 0.5),
                (Math.random() - 0.5),
            ).normalize().lerp(mainDir, 0.3).normalize();

            for (let i = 1; i <= branchLen; i++) {
                const prev = branchPts[branchPts.length - 1];
                const step = 0.12 + Math.random() * 0.15;
                const jit = new THREE.Vector3(
                    (Math.random() - 0.5) * this.jitter * 0.6,
                    (Math.random() - 0.5) * this.jitter * 0.6,
                    (Math.random() - 0.5) * this.jitter * 0.6,
                );
                const next = prev.clone().addScaledVector(branchDir, step).add(jit);
                branchPts.push(next);
            }
            branches.push(branchPts);
        }
        return branches;
    }

    /** Returns true if bolt has expired */
    isExpired(dt) {
        this.age += dt;
        return this.age >= this.maxAge;
    }

    /** Current alpha with flicker effect */
    getAlpha() {
        const lifeRatio = this.age / this.maxAge;
        const fadeIn = Math.min(1, lifeRatio * 10);         // first 10% ramp up
        const fadeOut = Math.max(0, 1 - (lifeRatio - 0.6) / 0.4); // last 40% fade
        const flicker = 0.65 + Math.random() * 0.35;        // per-frame jitter
        return fadeIn * fadeOut * flicker;
    }

    /** Total number of line segments (main + branches) */
    get totalSegments() {
        let n = Math.max(0, this.points.length - 1);
        for (const branch of this.branchPoints) {
            n += Math.max(0, branch.length - 1);
        }
        return n;
    }
}

// ── Synapse pathway system ──────────────────────────────────────────

export class BrainSynapses {
    constructor(scene, regions) {
        this.scene = scene;
        this.regions = regions;
        this.pathways = [];
        this.curves = {};

        this.firingRates = {};
        this.neuromodulation = { da: 1.2, ne: 1.2, serotonin: 1.0 };
        this._spawnTimers = {};

        this.activeBolts = [];

        this._buildPathways();
        this._buildLightningSystem();
    }

    _buildPathways() {
        const lineMat = new THREE.LineBasicMaterial({
            color: 0x2a3a5a,
            transparent: true,
            opacity: 0.12,
            linewidth: 1,
            depthWrite: false,
        });

        for (const pathway of SYNAPSE_PATHWAYS) {
            const srcCenter = this.regions.getCenter(pathway.src);
            const dstCenter = this.regions.getCenter(pathway.dst);

            // Control point: midpoint lifted/shifted for arc
            const mid = new THREE.Vector3().lerpVectors(srcCenter, dstCenter, 0.5);
            const dir = new THREE.Vector3().subVectors(dstCenter, srcCenter);
            const perp = new THREE.Vector3(-dir.z, dir.length() * 0.3, dir.x).normalize();
            mid.add(perp.multiplyScalar(dir.length() * 0.2));

            const curve = new THREE.QuadraticBezierCurve3(
                srcCenter.clone(),
                mid,
                dstCenter.clone(),
            );
            this.curves[pathway.id] = curve;

            // Faint static pathway line
            const points = curve.getPoints(20);
            const geo = new THREE.BufferGeometry().setFromPoints(points);
            const line = new THREE.Line(geo, lineMat.clone());
            this.scene.add(line);

            this.pathways.push({
                ...pathway,
                line,
                curve,
            });

            this._spawnTimers[pathway.id] = 0;
        }
    }

    _buildLightningSystem() {
        // Core: bright white-hot center lines
        this.corePositions = new Float32Array(MAX_BOLT_SEGMENTS * 6);
        this.coreColors = new Float32Array(MAX_BOLT_SEGMENTS * 6);

        const coreGeo = new THREE.BufferGeometry();
        coreGeo.setAttribute('position', new THREE.BufferAttribute(this.corePositions, 3));
        coreGeo.setAttribute('color', new THREE.BufferAttribute(this.coreColors, 3));
        coreGeo.setDrawRange(0, 0);

        const coreMat = new THREE.LineBasicMaterial({
            vertexColors: true,
            transparent: true,
            opacity: 0.9,
            blending: THREE.AdditiveBlending,
            depthWrite: false,
        });

        this.coreMesh = new THREE.LineSegments(coreGeo, coreMat);
        this.coreMesh.frustumCulled = false;
        this.scene.add(this.coreMesh);

        // Glow: dimmer region-colored lines (bloom amplifies these)
        this.glowPositions = new Float32Array(MAX_BOLT_SEGMENTS * 6);
        this.glowColors = new Float32Array(MAX_BOLT_SEGMENTS * 6);

        const glowGeo = new THREE.BufferGeometry();
        glowGeo.setAttribute('position', new THREE.BufferAttribute(this.glowPositions, 3));
        glowGeo.setAttribute('color', new THREE.BufferAttribute(this.glowColors, 3));
        glowGeo.setDrawRange(0, 0);

        const glowMat = new THREE.LineBasicMaterial({
            vertexColors: true,
            transparent: true,
            opacity: 0.35,
            blending: THREE.AdditiveBlending,
            depthWrite: false,
        });

        this.glowMesh = new THREE.LineSegments(glowGeo, glowMat);
        this.glowMesh.frustumCulled = false;
        this.scene.add(this.glowMesh);
    }

    setFiringRates(firingRates) {
        this.firingRates = firingRates;
    }

    setNeuromodulation(nm) {
        if (nm) this.neuromodulation = nm;
    }

    update(dt) {
        this._spawnBolts(dt);
        this._updateBolts(dt);
    }

    _spawnBolts(dt) {
        const da = this.neuromodulation.da || 1.2;
        const serotonin = this.neuromodulation.serotonin || 1.0;
        const serotoninDampen = Math.max(0.3, 1.0 - (serotonin - 0.5) * 0.3);

        for (const pathway of this.pathways) {
            this._spawnTimers[pathway.id] -= dt;
            if (this._spawnTimers[pathway.id] > 0) continue;

            const srcRate = this.firingRates[pathway.src] || 0;
            if (srcRate < 0.005) continue;

            const spawnRate = srcRate * 40 * serotoninDampen;
            if (Math.random() > spawnRate * dt * 10) continue;

            // Region color modulated by DA
            const regionInfo = this.regions.regionData[pathway.src];
            const baseColor = regionInfo
                ? regionInfo.baseColor.clone()
                : new THREE.Color(0.5, 0.5, 0.8);
            const saturation = 0.5 + Math.min(1.0, da * 0.4);
            baseColor.multiplyScalar(saturation);

            const intensity = Math.min(1.0, Math.sqrt(srcRate * 5.0));
            // Cap active bolts to prevent visual overload
            if (this.activeBolts.length >= 60) continue;

            const bolt = new LightningBolt(pathway.curve, baseColor, intensity);
            this.activeBolts.push(bolt);

            this._spawnTimers[pathway.id] = SPAWN_INTERVAL + (1 - intensity) * 0.15;
        }
    }

    _updateBolts(dt) {
        // Remove expired bolts
        this.activeBolts = this.activeBolts.filter(b => !b.isExpired(dt));

        let segIdx = 0;
        const maxSegs = MAX_BOLT_SEGMENTS;

        for (const bolt of this.activeBolts) {
            const alpha = bolt.getAlpha();
            const color = bolt.color;

            // ── Main bolt segments ──
            const pts = bolt.points;
            for (let i = 0; i < pts.length - 1 && segIdx < maxSegs; i++) {
                const a = pts[i];
                const b = pts[i + 1];
                const off = segIdx * 6;

                // Core: bright white-hot center (high bloom feed)
                const coreR = Math.min(1.5, (color.r * 0.2 + 0.8) * alpha * 1.4);
                const coreG = Math.min(1.5, (color.g * 0.2 + 0.8) * alpha * 1.4);
                const coreB = Math.min(1.5, (color.b * 0.15 + 0.85) * alpha * 1.4);

                this.corePositions[off]     = a.x;
                this.corePositions[off + 1] = a.y;
                this.corePositions[off + 2] = a.z;
                this.corePositions[off + 3] = b.x;
                this.corePositions[off + 4] = b.y;
                this.corePositions[off + 5] = b.z;

                this.coreColors[off]     = coreR;
                this.coreColors[off + 1] = coreG;
                this.coreColors[off + 2] = coreB;
                this.coreColors[off + 3] = coreR;
                this.coreColors[off + 4] = coreG;
                this.coreColors[off + 5] = coreB;

                // Glow: region-colored corona (bloom amplifies)
                const glowR = color.r * alpha * 1.2;
                const glowG = color.g * alpha * 1.2;
                const glowB = color.b * alpha * 1.2;

                this.glowPositions[off]     = a.x;
                this.glowPositions[off + 1] = a.y;
                this.glowPositions[off + 2] = a.z;
                this.glowPositions[off + 3] = b.x;
                this.glowPositions[off + 4] = b.y;
                this.glowPositions[off + 5] = b.z;

                this.glowColors[off]     = glowR;
                this.glowColors[off + 1] = glowG;
                this.glowColors[off + 2] = glowB;
                this.glowColors[off + 3] = glowR;
                this.glowColors[off + 4] = glowG;
                this.glowColors[off + 5] = glowB;

                segIdx++;
            }

            // ── Branch segments (dimmer) ──
            for (const branch of bolt.branchPoints) {
                for (let i = 0; i < branch.length - 1 && segIdx < maxSegs; i++) {
                    const a = branch[i];
                    const b = branch[i + 1];
                    const off = segIdx * 6;
                    const branchAlpha = alpha * 0.45;

                    const cR = Math.min(1, color.r * 0.25 + 0.75) * branchAlpha;
                    const cG = Math.min(1, color.g * 0.25 + 0.75) * branchAlpha;
                    const cB = Math.min(1, color.b * 0.2 + 0.8) * branchAlpha;

                    this.corePositions[off]     = a.x;
                    this.corePositions[off + 1] = a.y;
                    this.corePositions[off + 2] = a.z;
                    this.corePositions[off + 3] = b.x;
                    this.corePositions[off + 4] = b.y;
                    this.corePositions[off + 5] = b.z;

                    this.coreColors[off]     = cR;
                    this.coreColors[off + 1] = cG;
                    this.coreColors[off + 2] = cB;
                    this.coreColors[off + 3] = cR;
                    this.coreColors[off + 4] = cG;
                    this.coreColors[off + 5] = cB;

                    const gR = color.r * branchAlpha * 0.5;
                    const gG = color.g * branchAlpha * 0.5;
                    const gB = color.b * branchAlpha * 0.5;

                    this.glowPositions[off]     = a.x;
                    this.glowPositions[off + 1] = a.y;
                    this.glowPositions[off + 2] = a.z;
                    this.glowPositions[off + 3] = b.x;
                    this.glowPositions[off + 4] = b.y;
                    this.glowPositions[off + 5] = b.z;

                    this.glowColors[off]     = gR;
                    this.glowColors[off + 1] = gG;
                    this.glowColors[off + 2] = gB;
                    this.glowColors[off + 3] = gR;
                    this.glowColors[off + 4] = gG;
                    this.glowColors[off + 5] = gB;

                    segIdx++;
                }
            }
        }

        // Zero out unused segments
        for (let i = segIdx; i < maxSegs; i++) {
            const off = i * 6;
            this.corePositions[off + 1] = -100;
            this.corePositions[off + 4] = -100;
            this.coreColors[off] = 0; this.coreColors[off + 1] = 0; this.coreColors[off + 2] = 0;
            this.coreColors[off + 3] = 0; this.coreColors[off + 4] = 0; this.coreColors[off + 5] = 0;
            this.glowPositions[off + 1] = -100;
            this.glowPositions[off + 4] = -100;
            this.glowColors[off] = 0; this.glowColors[off + 1] = 0; this.glowColors[off + 2] = 0;
            this.glowColors[off + 3] = 0; this.glowColors[off + 4] = 0; this.glowColors[off + 5] = 0;
        }

        // Update draw range and flag dirty
        const vertexCount = segIdx * 2;
        this.coreMesh.geometry.setDrawRange(0, vertexCount);
        this.glowMesh.geometry.setDrawRange(0, vertexCount);

        this.coreMesh.geometry.attributes.position.needsUpdate = true;
        this.coreMesh.geometry.attributes.color.needsUpdate = true;
        this.glowMesh.geometry.attributes.position.needsUpdate = true;
        this.glowMesh.geometry.attributes.color.needsUpdate = true;
    }

    dispose() {
        this.coreMesh.geometry.dispose();
        this.coreMesh.material.dispose();
        this.scene.remove(this.coreMesh);
        this.glowMesh.geometry.dispose();
        this.glowMesh.material.dispose();
        this.scene.remove(this.glowMesh);
        for (const p of this.pathways) {
            p.line.geometry.dispose();
            p.line.material.dispose();
            this.scene.remove(p.line);
        }
    }
}
