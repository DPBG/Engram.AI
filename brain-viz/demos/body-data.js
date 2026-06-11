/**
 * body-data.js — WebSocket bridge for MuJoCo body state.
 *
 * Connects to the dashboard WebSocket, listens for "mujoco_state"
 * messages, and notifies listeners.  Falls back to a gentle idle
 * animation when no live data arrives.
 *
 * Usage:
 *   import { BodyData } from './body-data.js';
 *   const data = new BodyData();
 *   data.onUpdate = (state) => { ... };
 *   data.start('ws://host:8080/ws', 'http://host:8080');
 */

// Default standing pose (22 bodies) — 29-DOF humanoid (Optimus/G1/Atlas class)
// MuJoCo coords: X=right, Y=forward, Z=up.  Pelvis at 0.92m.
const STANDING_BODIES = [
    { name: 'world',          xpos: [0, 0, 0],        xquat: [1, 0, 0, 0] },
    // Core
    { name: 'pelvis',         xpos: [0, 0, 0.92],     xquat: [1, 0, 0, 0] },
    { name: 'lower_torso',    xpos: [0, 0, 1.00],     xquat: [1, 0, 0, 0] },
    { name: 'torso',          xpos: [0, 0, 1.18],     xquat: [1, 0, 0, 0] },
    // Head
    { name: 'neck_base',      xpos: [0, 0, 1.33],     xquat: [1, 0, 0, 0] },
    { name: 'head',           xpos: [0, 0, 1.41],     xquat: [1, 0, 0, 0] },
    // Right arm
    { name: 'r_shoulder_link', xpos: [0.18, 0, 1.26], xquat: [1, 0, 0, 0] },
    { name: 'r_upper_arm',    xpos: [0.18, 0, 1.11],  xquat: [1, 0, 0, 0] },
    { name: 'r_forearm',      xpos: [0.18, 0, 0.84],  xquat: [1, 0, 0, 0] },
    { name: 'r_hand',         xpos: [0.18, 0, 0.60],  xquat: [1, 0, 0, 0] },
    // Left arm
    { name: 'l_shoulder_link', xpos: [-0.18, 0, 1.26], xquat: [1, 0, 0, 0] },
    { name: 'l_upper_arm',    xpos: [-0.18, 0, 1.11], xquat: [1, 0, 0, 0] },
    { name: 'l_forearm',      xpos: [-0.18, 0, 0.84], xquat: [1, 0, 0, 0] },
    { name: 'l_hand',         xpos: [-0.18, 0, 0.60], xquat: [1, 0, 0, 0] },
    // Right leg
    { name: 'r_hip_link',     xpos: [0.09, 0, 0.86],  xquat: [1, 0, 0, 0] },
    { name: 'r_thigh',        xpos: [0.09, 0, 0.66],  xquat: [1, 0, 0, 0] },
    { name: 'r_shin',         xpos: [0.09, 0, 0.30],  xquat: [1, 0, 0, 0] },
    { name: 'r_foot',         xpos: [0.09, 0, 0.015], xquat: [1, 0, 0, 0] },
    // Left leg
    { name: 'l_hip_link',     xpos: [-0.09, 0, 0.86], xquat: [1, 0, 0, 0] },
    { name: 'l_thigh',        xpos: [-0.09, 0, 0.66], xquat: [1, 0, 0, 0] },
    { name: 'l_shin',         xpos: [-0.09, 0, 0.30], xquat: [1, 0, 0, 0] },
    { name: 'l_foot',         xpos: [-0.09, 0, 0.015], xquat: [1, 0, 0, 0] },
];

export class BodyData {
    constructor() {
        this.onUpdate = null;     // (state) => void
        this.onStatus = null;     // (statusObj) => void
        this.onTestPulse = null;  // ({channel, success}) => void
        this.onSelfView = null;   // ({pixels, width, height}) => void — body camera frame
        this._ws = null;
        this._simInterval = null;
        this._lastLiveTs = 0;
        this._reconnectTimer = null;
        this._wsUrl = null;
        this._apiBase = null;
        this._modelGeoms = null;
        this._liveCount = 0;

        this._pauseSimUntil = 0;  // timestamp — suppress sim fallback until

        // Exposed stats
        this.connected = false;
        this.mode = 'idle';       // 'live' | 'simulated' | 'idle'
    }

    async start(wsUrl, apiBase) {
        this._wsUrl = wsUrl;
        this._apiBase = apiBase;
        await this._fetchModel();
        this._connectWs();
        this._startSimFallback();
    }

    stop() {
        if (this._ws) { this._ws.close(); this._ws = null; }
        if (this._simInterval) { clearInterval(this._simInterval); this._simInterval = null; }
        if (this._reconnectTimer) { clearTimeout(this._reconnectTimer); this._reconnectTimer = null; }
    }

    getModelGeoms() {
        return this._modelGeoms;
    }

    // ── Model geometry (one-time fetch) ──────────────────────────

    async _fetchModel() {
        try {
            const resp = await fetch(this._apiBase + '/api/mujoco/model');
            const json = await resp.json();
            this._modelGeoms = json.geoms || [];
        } catch (e) {
            console.warn('Failed to fetch MuJoCo model, using defaults:', e);
            this._modelGeoms = null;
        }
    }

    // ── WebSocket connection ─────────────────────────────────────

    _connectWs() {
        if (this._ws) return;
        try {
            this._ws = new WebSocket(this._wsUrl);
        } catch (e) {
            this._scheduleReconnect();
            return;
        }

        this._ws.onopen = () => {
            this.connected = true;
            this._emitStatus();
        };

        this._ws.onmessage = (evt) => {
            try {
                const msg = JSON.parse(evt.data);
                if (msg.type === 'mujoco_state' && msg.data) {
                    this._handleLiveState(msg.data);
                } else if (msg.type === 'visual_body_frame' && msg.data) {
                    this._handleSelfView(msg.data);
                }
            } catch (e) { /* ignore parse errors */ }
        };

        this._ws.onclose = () => {
            this.connected = false;
            this._ws = null;
            this._emitStatus();
            this._scheduleReconnect();
        };

        this._ws.onerror = () => {
            // onclose will fire after this
        };
    }

    _scheduleReconnect() {
        if (this._reconnectTimer) return;
        this._reconnectTimer = setTimeout(() => {
            this._reconnectTimer = null;
            this._connectWs();
        }, 3000);
    }

    _handleLiveState(data) {
        this._lastLiveTs = Date.now();
        // Hold the live pose for at least 5s (prevents sim fallback overwrite)
        this._pauseSimUntil = Math.max(this._pauseSimUntil, Date.now() + 5000);
        this._liveCount++;
        this.mode = 'live';

        // Derive per-channel motor activation from joint velocities.
        // qvel layout: [0-5]=root freejoint (6 DOF), then 29 hinge joints:
        //   [6-8]=waist(loco), [9-10]=neck(head), [11-16]=r_arm(manip),
        //   [17-22]=l_arm(manip), [23-28]=r_leg(loco), [29-34]=l_leg(loco)
        let motorRates = null;
        const jv = data.joint_velocities;
        if (jv && jv.length >= 35) {  // 6 freejoint + 29 hinge = 35
            const absSum = (start, end) => {
                let s = 0;
                for (let i = start; i <= end; i++) s += Math.abs(jv[i]);
                return s;
            };
            // Normalize by joint count, then clamp to [0,1] (scale ~2 rad/s = full glow)
            const locoRaw = (absSum(6, 8) + absSum(23, 34)) / 15;
            const manipRaw = absSum(11, 22) / 12;
            const headRaw = absSum(9, 10) / 2;
            const scale = 0.5;  // rad/s → [0,1] mapping
            motorRates = {
                locomotion: Math.min(locoRaw * scale, 1.0),
                manipulation: Math.min(manipRaw * scale, 1.0),
                head: Math.min(headRaw * scale, 1.0),
            };
        }

        if (this.onUpdate) {
            this.onUpdate({
                bodies: data.bodies || [],
                activeChannel: data.active_channel || null,
                success: data.success || false,
                torsoHeight: data.torso_height || 0,
                simTime: data.sim_time || 0,
                numContacts: data.num_contacts || 0,
                jointVelocities: data.joint_velocities || null,
                liveCount: this._liveCount,
                motorRates,
            });
        }
        this._emitStatus();
    }

    _handleSelfView(data) {
        if (!this.onSelfView || !data.pixels_b64) return;
        // Decode base64 → Uint8Array (64x64 grayscale)
        const raw = atob(data.pixels_b64);
        const pixels = new Uint8Array(raw.length);
        for (let i = 0; i < raw.length; i++) pixels[i] = raw.charCodeAt(i);
        this.onSelfView({ pixels, width: data.width || 64, height: data.height || 64 });
    }

    // ── Simulated fallback (idle sway when no live data) ─────────

    /**
     * Pause sim fallback for `holdMs` without triggering a test pulse.
     * Used by guide panel for instant client-side preview.
     */
    pauseSim(holdMs = 15000) {
        this._pauseSimUntil = Date.now() + holdMs;
        this.mode = 'simulated';
        this._emitStatus();
    }

    /**
     * Inject a test pulse — moves the body for `holdMs` then reverts to idle.
     */
    sendTestPulse(channel, holdMs = 10000) {
        this._pauseSimUntil = Date.now() + holdMs;
        this.mode = 'simulated';
        const success = Math.random() > 0.4;
        if (this.onTestPulse) {
            this.onTestPulse({ channel, success });
        }
        // Also update HUD stats via onUpdate (but bodies will be set by scene directly)
        if (this.onUpdate) {
            this.onUpdate({
                bodies: null,  // scene handles pose directly
                activeChannel: channel,
                success,
                torsoHeight: 1.2,
                simTime: Date.now() / 1000,
                numContacts: 2,
                liveCount: this._liveCount,
            });
        }
        this._emitStatus();
    }

    _generateTestPose(channel) {
        const base = STANDING_BODIES.map(b => ({
            name: b.name,
            xpos: [...b.xpos],
            xquat: [...b.xquat],
        }));
        // Look up bodies by name rather than fragile array indices
        const byName = {};
        base.forEach((b, i) => { byName[b.name] = i; });

        // MuJoCo coords: X=right, Y=forward, Z=up.  xquat = [w, x, y, z].
        if (channel === 'locomotion') {
            // Right leg forward
            const rt = byName['r_thigh'], rs = byName['r_shin'], rf = byName['r_foot'];
            if (rt !== undefined) { base[rt].xpos = [0.09, 0.12, 0.58]; base[rt].xquat = [0.966, 0, 0.259, 0]; }
            if (rs !== undefined) { base[rs].xpos = [0.09, 0.28, 0.26]; base[rs].xquat = [0.94, 0, 0.342, 0]; }
            if (rf !== undefined) { base[rf].xpos = [0.09, 0.32, 0.015]; }
            // Left leg back
            const lt = byName['l_thigh'], ls = byName['l_shin'], lf = byName['l_foot'];
            if (lt !== undefined) { base[lt].xpos = [-0.09, -0.06, 0.62]; base[lt].xquat = [0.991, 0, -0.131, 0]; }
            if (ls !== undefined) { base[ls].xpos = [-0.09, -0.10, 0.28]; base[ls].xquat = [0.991, 0, -0.131, 0]; }
            if (lf !== undefined) { base[lf].xpos = [-0.09, -0.12, 0.015]; }
        } else if (channel === 'manipulation') {
            const rArmQ = [0.65, 0.27, 0.65, 0.27];  // out + forward
            const rForeQ = [0.56, 0.33, 0.56, 0.33];
            const lArmQ = [0.65, 0.27, -0.65, -0.27]; // mirrored
            const lForeQ = [0.56, 0.33, -0.56, -0.33];
            const ru = byName['r_upper_arm'], rfa = byName['r_forearm'], rh = byName['r_hand'];
            if (ru !== undefined) { base[ru].xpos = [0.30, 0.1, 1.18]; base[ru].xquat = rArmQ; }
            if (rfa !== undefined) { base[rfa].xpos = [0.40, 0.25, 1.10]; base[rfa].xquat = rForeQ; }
            if (rh !== undefined) { base[rh].xpos = [0.48, 0.36, 1.02]; base[rh].xquat = rForeQ; }
            const lu = byName['l_upper_arm'], lfa = byName['l_forearm'], lh = byName['l_hand'];
            if (lu !== undefined) { base[lu].xpos = [-0.30, 0.1, 1.18]; base[lu].xquat = lArmQ; }
            if (lfa !== undefined) { base[lfa].xpos = [-0.40, 0.25, 1.10]; base[lfa].xquat = lForeQ; }
            if (lh !== undefined) { base[lh].xpos = [-0.48, 0.36, 1.02]; base[lh].xquat = lForeQ; }
        } else if (channel === 'head') {
            const hi = byName['head'];
            if (hi !== undefined) { base[hi].xpos = [0.02, 0, 1.43]; base[hi].xquat = [0.976, 0.1, 0.05, 0.19]; }
        }
        return base;
    }

    _startSimFallback() {
        this._simInterval = setInterval(() => {
            // If live data arrived in last 5s, skip simulation
            if (Date.now() - this._lastLiveTs < 5000) return;
            // If a test pulse is active, don't overwrite it
            if (Date.now() < this._pauseSimUntil) return;

            this.mode = this.connected ? 'simulated' : 'idle';
            const t = Date.now() / 1000;
            const bodies = this._generateIdlePose(t);

            if (this.onUpdate) {
                this.onUpdate({
                    bodies,
                    activeChannel: null,
                    success: false,
                    torsoHeight: 1.2 + Math.sin(t * 0.5) * 0.01,
                    simTime: t,
                    numContacts: 2,
                    liveCount: this._liveCount,
                });
            }
            this._emitStatus();
        }, 100);  // 10 Hz simulated updates
    }

    _generateIdlePose(t) {
        // Subtle breathing sway on the standing pose
        const sway = Math.sin(t * 0.5) * 0.008;
        return STANDING_BODIES.map(b => {
            const pos = [...b.xpos];
            if (b.name !== 'world') {
                pos[2] += sway;
            }
            return { name: b.name, xpos: pos, xquat: [...b.xquat] };
        });
    }

    _emitStatus() {
        if (this.onStatus) {
            this.onStatus({
                connected: this.connected,
                mode: this.mode,
                liveCount: this._liveCount,
            });
        }
    }
}
