/**
 * body-scene.js — Three.js humanoid robot scene.
 *
 * Procedurally builds a sleek humanoid robot with panel armor,
 * visible rotary joint actuators, and metallic materials.
 * Matches the 29-DOF MuJoCo body hierarchy.
 *
 * Usage:
 *   import { BodyScene } from './body-scene.js';
 *   const scene = new BodyScene(containerEl);
 *   scene.setModelGeometry(geomsArray);   // once, from /api/mujoco/model
 *   scene.updateBodies(bodiesArray, channel, success);  // per frame
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// ── Materials ──────────────────────────────────────────────────
const _white   = () => new THREE.MeshStandardMaterial({ color: 0xdce0e6, roughness: 0.35, metalness: 0.10 });
const _darkGrey = () => new THREE.MeshStandardMaterial({ color: 0x1a1d22, roughness: 0.45, metalness: 0.25 });
const _midGrey  = () => new THREE.MeshStandardMaterial({ color: 0x44484f, roughness: 0.40, metalness: 0.30 });
const _joint    = () => new THREE.MeshStandardMaterial({ color: 0x2a2d33, roughness: 0.30, metalness: 0.60 });
const _accent   = () => new THREE.MeshStandardMaterial({ color: 0x3a7bd5, roughness: 0.25, metalness: 0.40, emissive: 0x1a3a6a, emissiveIntensity: 0.15 });
const _eye      = () => new THREE.MeshStandardMaterial({ color: 0x44aaff, roughness: 0.1, metalness: 0.3, emissive: 0x2288ff, emissiveIntensity: 0.8 });
const _visor    = () => new THREE.MeshStandardMaterial({ color: 0x111418, roughness: 0.15, metalness: 0.60, transparent: true, opacity: 0.85 });

// Channel → highlight color mapping
const CHANNEL_COLORS = {
    locomotion:   new THREE.Color(0x22aaff),
    manipulation: new THREE.Color(0x38d97a),
    head:         new THREE.Color(0xf5c842),
    speech:       new THREE.Color(0xf472b6),
    expression:   new THREE.Color(0xa673f5),
    cognitive:    new THREE.Color(0x22d3ee),
};

// Which bodies belong to each channel (for highlight)
const CHANNEL_BODIES = {
    locomotion:   [
        'pelvis', 'lower_torso',
        'r_hip_link', 'r_thigh', 'r_shin', 'r_foot',
        'l_hip_link', 'l_thigh', 'l_shin', 'l_foot',
    ],
    manipulation: [
        'r_shoulder_link', 'r_upper_arm', 'r_forearm', 'r_hand',
        'l_shoulder_link', 'l_upper_arm', 'l_forearm', 'l_hand',
    ],
    head: ['neck_base', 'head'],
};

// ── Procedural robot body part builders ───────────────────────
// Each function returns a THREE.Group with child meshes.
// Dimensions tuned to match the MJCF proportions.

function _jointRing(radius, width) {
    const geo = new THREE.TorusGeometry(radius, width, 8, 24);
    const mesh = new THREE.Mesh(geo, _joint());
    mesh.rotation.x = Math.PI / 2;
    return mesh;
}

function _jointDisc(radius) {
    const geo = new THREE.CylinderGeometry(radius, radius, 0.015, 24);
    return new THREE.Mesh(geo, _joint());
}

function buildPelvis() {
    const g = new THREE.Group();
    // Main hip plate
    const plate = new THREE.Mesh(
        new THREE.BoxGeometry(0.26, 0.10, 0.16),
        _darkGrey(),
    );
    g.add(plate);
    // Side hip covers
    for (const s of [-1, 1]) {
        const cover = new THREE.Mesh(
            new THREE.BoxGeometry(0.04, 0.09, 0.13),
            _midGrey(),
        );
        cover.position.set(s * 0.12, -0.005, 0);
        g.add(cover);
    }
    return g;
}

function buildLowerTorso() {
    const g = new THREE.Group();
    // Waist cylinder (visible actuator housing)
    const waist = new THREE.Mesh(
        new THREE.CylinderGeometry(0.10, 0.11, 0.14, 16),
        _darkGrey(),
    );
    g.add(waist);
    // Ring detail
    g.add(_jointRing(0.105, 0.008));
    return g;
}

function buildTorso() {
    const g = new THREE.Group();
    // Chest plate — front panel (white)
    const chest = new THREE.Mesh(
        new THREE.BoxGeometry(0.26, 0.22, 0.14),
        _white(),
    );
    g.add(chest);
    // Center accent line
    const line = new THREE.Mesh(
        new THREE.BoxGeometry(0.005, 0.18, 0.145),
        _accent(),
    );
    g.add(line);
    // Dark side panels
    for (const s of [-1, 1]) {
        const side = new THREE.Mesh(
            new THREE.BoxGeometry(0.02, 0.20, 0.12),
            _darkGrey(),
        );
        side.position.set(s * 0.13, 0, 0);
        g.add(side);
    }
    return g;
}

function buildNeckBase() {
    const g = new THREE.Group();
    const neck = new THREE.Mesh(
        new THREE.CylinderGeometry(0.035, 0.045, 0.06, 12),
        _darkGrey(),
    );
    g.add(neck);
    g.add(_jointRing(0.04, 0.006));
    return g;
}

function buildHead() {
    const g = new THREE.Group();
    // Main head shell — rounded box via sphere segments
    const skull = new THREE.Mesh(
        new THREE.SphereGeometry(0.095, 24, 16, 0, Math.PI * 2, 0, Math.PI * 0.75),
        _white(),
    );
    skull.position.y = 0.02;
    g.add(skull);
    // Face plate / visor — faces -Z (MuJoCo forward)
    const visor = new THREE.Mesh(
        new THREE.SphereGeometry(0.088, 20, 8, -Math.PI * 0.35, Math.PI * 0.7, Math.PI * 0.15, Math.PI * 0.35),
        _visor(),
    );
    visor.position.set(0, 0.02, 0);
    // SphereGeometry phi=0 faces +Z by default; rotate 180 to face -Z
    visor.rotation.y = Math.PI;
    g.add(visor);
    // Eyes (emissive blue) — face -Z
    for (const s of [-1, 1]) {
        const eye = new THREE.Mesh(
            new THREE.SphereGeometry(0.015, 8, 6),
            _eye(),
        );
        eye.position.set(s * 0.032, 0.03, -0.075);
        g.add(eye);
    }
    // Chin plate — face -Z
    const chin = new THREE.Mesh(
        new THREE.BoxGeometry(0.08, 0.025, 0.04),
        _midGrey(),
    );
    chin.position.set(0, -0.05, -0.04);
    g.add(chin);
    return g;
}

function buildShoulderLink() {
    const g = new THREE.Group();
    // Shoulder actuator housing (visible rotary joint)
    const housing = new THREE.Mesh(
        new THREE.CylinderGeometry(0.042, 0.042, 0.05, 16),
        _joint(),
    );
    housing.rotation.z = Math.PI / 2;
    g.add(housing);
    // Outer disc
    const disc = _jointDisc(0.045);
    disc.rotation.z = Math.PI / 2;
    g.add(disc);
    return g;
}

function buildUpperArm() {
    const g = new THREE.Group();
    // Main arm segment — white panel
    const arm = new THREE.Mesh(
        new THREE.CylinderGeometry(0.038, 0.033, 0.24, 8),
        _white(),
    );
    arm.position.y = -0.12;
    g.add(arm);
    // Dark inner stripe
    const stripe = new THREE.Mesh(
        new THREE.BoxGeometry(0.015, 0.22, 0.065),
        _darkGrey(),
    );
    stripe.position.set(0, -0.12, 0);
    g.add(stripe);
    // Elbow joint disc at bottom
    const elbow = _jointDisc(0.035);
    elbow.position.y = -0.24;
    g.add(elbow);
    return g;
}

function buildForearm() {
    const g = new THREE.Group();
    const arm = new THREE.Mesh(
        new THREE.CylinderGeometry(0.032, 0.028, 0.22, 8),
        _white(),
    );
    arm.position.y = -0.11;
    g.add(arm);
    const stripe = new THREE.Mesh(
        new THREE.BoxGeometry(0.012, 0.20, 0.055),
        _darkGrey(),
    );
    stripe.position.set(0, -0.11, 0);
    g.add(stripe);
    // Wrist ring
    g.add(_jointRing(0.03, 0.005));
    const wr = g.children[g.children.length - 1];
    wr.position.y = -0.22;
    wr.rotation.x = Math.PI / 2;
    return g;
}

function buildHand() {
    const g = new THREE.Group();
    // Palm
    const palm = new THREE.Mesh(
        new THREE.BoxGeometry(0.06, 0.08, 0.025),
        _darkGrey(),
    );
    g.add(palm);
    // Finger segments (simplified 3 prongs)
    for (const x of [-0.018, 0, 0.018]) {
        const finger = new THREE.Mesh(
            new THREE.BoxGeometry(0.012, 0.045, 0.018),
            _midGrey(),
        );
        finger.position.set(x, -0.06, 0);
        g.add(finger);
    }
    // Thumb
    const thumb = new THREE.Mesh(
        new THREE.BoxGeometry(0.015, 0.035, 0.018),
        _midGrey(),
    );
    thumb.position.set(0.035, -0.02, 0);
    thumb.rotation.z = -0.5;
    g.add(thumb);
    return g;
}

function buildHipLink() {
    const g = new THREE.Group();
    const housing = new THREE.Mesh(
        new THREE.CylinderGeometry(0.048, 0.048, 0.04, 16),
        _joint(),
    );
    g.add(housing);
    const ring = _jointDisc(0.05);
    ring.position.y = 0.02;
    g.add(ring);
    return g;
}

function buildThigh() {
    const g = new THREE.Group();
    // Thigh armor panel (white)
    const thigh = new THREE.Mesh(
        new THREE.CylinderGeometry(0.055, 0.045, 0.34, 8),
        _white(),
    );
    thigh.position.y = -0.17;
    g.add(thigh);
    // Dark inner channel
    const channel = new THREE.Mesh(
        new THREE.BoxGeometry(0.018, 0.32, 0.09),
        _darkGrey(),
    );
    channel.position.set(0, -0.17, 0);
    g.add(channel);
    // Knee joint at bottom
    const knee = _jointDisc(0.043);
    knee.position.y = -0.34;
    g.add(knee);
    return g;
}

function buildShin() {
    const g = new THREE.Group();
    const shin = new THREE.Mesh(
        new THREE.CylinderGeometry(0.042, 0.035, 0.32, 8),
        _white(),
    );
    shin.position.y = -0.16;
    g.add(shin);
    const channel = new THREE.Mesh(
        new THREE.BoxGeometry(0.015, 0.30, 0.07),
        _darkGrey(),
    );
    channel.position.set(0, -0.16, 0);
    g.add(channel);
    // Ankle ring
    const ankle = _jointRing(0.033, 0.005);
    ankle.position.y = -0.32;
    ankle.rotation.x = Math.PI / 2;
    g.add(ankle);
    return g;
}

function buildFoot() {
    const g = new THREE.Group();
    const foot = new THREE.Mesh(
        new THREE.BoxGeometry(0.065, 0.03, 0.10),
        _darkGrey(),
    );
    foot.position.set(0, 0, -0.015);
    g.add(foot);
    // Sole accent
    const sole = new THREE.Mesh(
        new THREE.BoxGeometry(0.06, 0.008, 0.095),
        _midGrey(),
    );
    sole.position.set(0, -0.015, -0.015);
    g.add(sole);
    return g;
}

// Map body names to builder functions
const BODY_BUILDERS = {
    'pelvis':          buildPelvis,
    'lower_torso':     buildLowerTorso,
    'torso':           buildTorso,
    'neck_base':       buildNeckBase,
    'head':            buildHead,
    'r_shoulder_link': buildShoulderLink,
    'r_upper_arm':     buildUpperArm,
    'r_forearm':       buildForearm,
    'r_hand':          buildHand,
    'l_shoulder_link': buildShoulderLink,
    'l_upper_arm':     buildUpperArm,
    'l_forearm':       buildForearm,
    'l_hand':          buildHand,
    'r_hip_link':      buildHipLink,
    'r_thigh':         buildThigh,
    'r_shin':          buildShin,
    'r_foot':          buildFoot,
    'l_hip_link':      buildHipLink,
    'l_thigh':         buildThigh,
    'l_shin':          buildShin,
    'l_foot':          buildFoot,
};

// ── Scene class ───────────────────────────────────────────────

export class BodyScene {
    constructor(container) {
        this._container = container;
        this._meshes = {};          // body name → THREE.Group
        this._baseMaterials = {};   // body name → [MeshStandardMaterial, ...]
        this._origEmissive = new Map(); // mat → { color: Color, intensity: number }
        this._activeChannel = null;
        this._animId = null;
        this._cameraTarget = new THREE.Vector3(0, 0.75, 0); // smooth-follow target

        this._initRenderer();
        this._initCamera();
        this._initScene();
        this._initControls();
        this._onResize = this._handleResize.bind(this);
        window.addEventListener('resize', this._onResize);
        this._animate();
    }

    // ── Setup ────────────────────────────────────────────────────

    _initRenderer() {
        this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.renderer.setSize(this._container.clientWidth, this._container.clientHeight);
        this.renderer.setClearColor(0x0a0e1a, 1);
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
        this.renderer.toneMappingExposure = 1.8;
        this._container.appendChild(this.renderer.domElement);
    }

    _initCamera() {
        const w = this._container.clientWidth;
        const h = this._container.clientHeight;
        this.camera = new THREE.PerspectiveCamera(40, w / h, 0.05, 50);
        this.camera.position.set(1.0, 1.1, -2.0);
        this.camera.lookAt(0, 0.75, 0);
    }

    _initScene() {
        this.scene = new THREE.Scene();
        this.scene.fog = new THREE.FogExp2(0x0a0e1a, 0.04);

        // Ambient
        const ambient = new THREE.AmbientLight(0x556677, 1.0);
        this.scene.add(ambient);

        // Key light — warm, strong (in front of robot at -Z)
        const key = new THREE.DirectionalLight(0xfff5e6, 1.6);
        key.position.set(3, 6, -4);
        key.castShadow = true;
        key.shadow.mapSize.set(2048, 2048);
        key.shadow.camera.near = 0.5;
        key.shadow.camera.far = 20;
        key.shadow.camera.left = -3;
        key.shadow.camera.right = 3;
        key.shadow.camera.top = 4;
        key.shadow.camera.bottom = -2;
        key.shadow.bias = -0.0005;
        this.scene.add(key);

        // Fill light — cool blue (behind robot)
        const fill = new THREE.DirectionalLight(0x6688cc, 0.5);
        fill.position.set(-3, 4, 2);
        this.scene.add(fill);

        // Rim light — highlights edges (behind robot)
        const rim = new THREE.DirectionalLight(0x88aaee, 0.4);
        rim.position.set(0, 2, 4);
        this.scene.add(rim);

        // Ground grid
        const grid = new THREE.GridHelper(10, 30, 0x1a2a3a, 0x0f1520);
        this.scene.add(grid);

        // Ground plane (shadow receiver)
        const ground = new THREE.Mesh(
            new THREE.PlaneGeometry(10, 10),
            new THREE.ShadowMaterial({ opacity: 0.4 }),
        );
        ground.rotation.x = -Math.PI / 2;
        ground.receiveShadow = true;
        this.scene.add(ground);
    }

    _initControls() {
        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.target.set(0, 0.75, 0);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.08;
        this.controls.minDistance = 0.8;
        this.controls.maxDistance = 8;
        this.controls.update();
    }

    // ── Build robot meshes ────────────────────────────────────────

    setModelGeometry(geoms) {
        // Remove old meshes
        for (const m of Object.values(this._meshes)) {
            this.scene.remove(m);
        }
        this._meshes = {};
        this._baseMaterials = {};

        // Build robot parts procedurally instead of using MJCF geoms
        this._buildRobot();
    }

    _buildRobot() {
        for (const [bodyName, builder] of Object.entries(BODY_BUILDERS)) {
            const group = builder();
            group.castShadow = true;
            group.traverse(child => {
                if (child.isMesh) {
                    child.castShadow = true;
                    child.receiveShadow = true;
                }
            });
            this.scene.add(group);
            this._meshes[bodyName] = group;

            // Collect materials for highlight and save original emissive values
            const mats = [];
            group.traverse(child => {
                if (child.isMesh && child.material) {
                    mats.push(child.material);
                    const m = child.material;
                    if (m.emissive) {
                        this._origEmissive.set(m, {
                            color: m.emissive.clone(),
                            intensity: m.emissiveIntensity,
                        });
                    }
                }
            });
            this._baseMaterials[bodyName] = mats;
        }
    }

    // ── Update body transforms from live data ────────────────────

    updateBodies(bodies, activeChannel, success) {
        if (!bodies) return;
        this._activeChannel = activeChannel;

        // First pass: find torso/pelvis to compute ground offset
        let torsoRawY = 0;
        for (const b of bodies) {
            if (b.name === 'torso' || b.name === 'pelvis') {
                torsoRawY = b.xpos[2];  // MuJoCo Z → Three.js Y
                break;
            }
        }

        // Clamp display: if body is above 3m, shift everything down so torso
        // stays at ~1m (standing height). Keeps body near the ground grid
        // instead of floating in empty space at 100m+.
        const MAX_DISPLAY_HEIGHT = 3.0;
        const yOffset = torsoRawY > MAX_DISPLAY_HEIGHT ? torsoRawY - 1.0 : 0;

        for (const b of bodies) {
            const mesh = this._meshes[b.name];
            if (!mesh) continue;

            // MuJoCo xpos → Three.js position (swap Y/Z, negate Y)
            // MuJoCo: X=right, Y=forward, Z=up → Three.js: X=right, Y=up, Z=-forward
            mesh.position.set(b.xpos[0], b.xpos[2] - yOffset, -b.xpos[1]);

            // MuJoCo xquat [w,x,y,z] → Three.js quaternion
            const [w, qx, qy, qz] = b.xquat;
            mesh.quaternion.set(qx, qz, -qy, w);
        }

        // Smooth camera follow using clamped position
        const clampedY = Math.max(0.3, torsoRawY - yOffset);
        this._cameraTarget.set(0, clampedY, 0);
        this.controls.target.lerp(this._cameraTarget, 0.08);

        this._highlightChannel(activeChannel, success);
    }

    /**
     * Apply a canned test pose directly in Three.js space.
     */
    applyTestPose(channel, success) {
        // Forward = -Z in Three.js (matches MuJoCo Y+ after coordinate swap)
        const poses = {
            locomotion: {
                r_thigh:  { pos: [0.09, 0.72, -0.12], rot: [0.26, 0, 0, 0.97] },
                r_shin:   { pos: [0.09, 0.36, -0.25], rot: [0.17, 0, 0, 0.98] },
                r_foot:   { pos: [0.09, 0.02, -0.28], rot: [0, 0, 0, 1] },
                l_thigh:  { pos: [-0.09, 0.78, 0.08], rot: [-0.13, 0, 0, 0.99] },
                l_shin:   { pos: [-0.09, 0.42, 0.12], rot: [-0.09, 0, 0, 1.0] },
                l_foot:   { pos: [-0.09, 0.02, 0.10], rot: [0, 0, 0, 1] },
            },
            manipulation: {
                r_upper_arm: { pos: [0.22, 1.24, -0.15], rot: [0.707, 0, 0, 0.707] },
                r_forearm:   { pos: [0.24, 1.20, -0.42], rot: [0.74, 0, 0, 0.67] },
                r_hand:      { pos: [0.26, 1.16, -0.66], rot: [0.74, 0, 0, 0.67] },
                l_upper_arm: { pos: [-0.22, 1.24, -0.15], rot: [0.707, 0, 0, 0.707] },
                l_forearm:   { pos: [-0.24, 1.20, -0.42], rot: [0.74, 0, 0, 0.67] },
                l_hand:      { pos: [-0.26, 1.16, -0.66], rot: [0.74, 0, 0, 0.67] },
            },
            head: {
                head: { pos: [0, 1.45, -0.03], rot: [0.08, 0.12, 0, 0.99] },
            },
        };

        const pose = poses[channel];
        if (!pose) return;

        for (const [name, t] of Object.entries(pose)) {
            const mesh = this._meshes[name];
            if (!mesh) continue;
            mesh.position.set(t.pos[0], t.pos[1], t.pos[2]);
            mesh.quaternion.set(t.rot[0], t.rot[1], t.rot[2], t.rot[3]).normalize();
        }

        this._highlightChannel(channel, success);
    }

    _highlightChannel(channel, success) {
        // Restore all materials to original emissive (preserves eye glow, accent, etc.)
        for (const mats of Object.values(this._baseMaterials)) {
            for (const mat of mats) {
                const orig = this._origEmissive.get(mat);
                if (orig) {
                    mat.emissive.copy(orig.color);
                    mat.emissiveIntensity = orig.intensity;
                } else if (mat.emissive) {
                    mat.emissive.setHex(0x000000);
                    mat.emissiveIntensity = 0;
                }
            }
        }

        if (!channel) return;
        const bodyNames = CHANNEL_BODIES[channel];
        if (!bodyNames) return;

        const highlightColor = CHANNEL_COLORS[channel] || new THREE.Color(0xffffff);
        const intensity = success ? 0.35 : 0.15;

        for (const name of bodyNames) {
            const mats = this._baseMaterials[name];
            if (!mats) continue;
            for (const mat of mats) {
                mat.emissive.copy(highlightColor);
                mat.emissiveIntensity = intensity;
            }
        }
    }

    /**
     * Apply per-channel motor activation as continuous glow intensity.
     * Only modifies materials on channel-mapped bodies (preserves eye/accent emissive).
     * @param {Object} rates — { locomotion: 0-1, manipulation: 0-1, head: 0-1 }
     */
    updateMotorActivation(rates) {
        if (!rates) return;
        // Collect all body names that belong to any channel
        const channelBodies = new Set();
        for (const names of Object.values(CHANNEL_BODIES)) {
            for (const n of names) channelBodies.add(n);
        }
        // Reset channel-mapped bodies to their original emissive (preserves eye glow, etc.)
        for (const name of channelBodies) {
            const mats = this._baseMaterials[name];
            if (!mats) continue;
            for (const mat of mats) {
                const orig = this._origEmissive.get(mat);
                if (orig) {
                    mat.emissive.copy(orig.color);
                    mat.emissiveIntensity = orig.intensity;
                } else if (mat.emissive) {
                    mat.emissive.setHex(0x000000);
                    mat.emissiveIntensity = 0;
                }
            }
        }
        for (const [channel, intensity] of Object.entries(rates)) {
            if (intensity < 0.01) continue;
            const bodyNames = CHANNEL_BODIES[channel];
            const color = CHANNEL_COLORS[channel];
            if (!bodyNames || !color) continue;
            const glow = Math.min(intensity, 1.0) * 0.5;
            for (const name of bodyNames) {
                const mats = this._baseMaterials[name];
                if (!mats) continue;
                for (const mat of mats) {
                    mat.emissive.copy(color);
                    mat.emissiveIntensity = Math.max(mat.emissiveIntensity, glow);
                }
            }
        }
    }

    // ── Animation loop ───────────────────────────────────────────

    _animate() {
        this._animId = requestAnimationFrame(() => this._animate());
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }

    _handleResize() {
        const w = this._container.clientWidth;
        const h = this._container.clientHeight;
        this.camera.aspect = w / h;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(w, h);
    }

    dispose() {
        window.removeEventListener('resize', this._onResize);
        if (this._animId) cancelAnimationFrame(this._animId);
        this.controls.dispose();
        this.renderer.dispose();
    }
}
