/**
 * body-guide.js — Teaching controls for the MuJoCo body.
 *
 * Provides UI for human-in-the-loop motor teaching:
 *   - Joint sliders (guide to pose)
 *   - Push buttons (apply external force)
 *   - Reward buttons (good/bad feedback)
 *   - Preset poses (stand, step, reach)
 *
 * Sends commands via WebSocket: { type: "mujoco_guide", data: {...} }
 *
 * Usage:
 *   import { BodyGuide } from './body-guide.js';
 *   const guide = new BodyGuide(containerEl, wsUrl, apiBase);
 *   guide.start();
 */

// All 29 joints — realistic humanoid (Optimus/G1/Atlas class).
// Every preset specifies ALL joints to prevent state leaking between poses.
const _ZERO = {
    // Waist (3)
    waist_yaw: 0, waist_roll: 0, waist_pitch: 0,
    // Neck (2)
    neck_pitch: 0, neck_yaw: 0,
    // Right arm (6)
    r_shoulder_pitch: 0, r_shoulder_roll: 0, r_shoulder_yaw: 0,
    r_elbow: 0, r_wrist_pitch: 0, r_wrist_yaw: 0,
    // Left arm (6)
    l_shoulder_pitch: 0, l_shoulder_roll: 0, l_shoulder_yaw: 0,
    l_elbow: 0, l_wrist_pitch: 0, l_wrist_yaw: 0,
    // Right leg (6)
    r_hip_yaw: 0, r_hip_roll: 0, r_hip_pitch: 0,
    r_knee: 0, r_ankle_pitch: 0, r_ankle_roll: 0,
    // Left leg (6)
    l_hip_yaw: 0, l_hip_roll: 0, l_hip_pitch: 0,
    l_knee: 0, l_ankle_pitch: 0, l_ankle_roll: 0,
};

const PRESETS = {
    stand: {
        label: 'Stand',
        joints: { ..._ZERO },
    },
    step_right: {
        label: 'Step R',
        joints: { ..._ZERO, r_hip_pitch: 25, r_knee: -15, r_ankle_pitch: 10, l_hip_pitch: -5 },
    },
    step_left: {
        label: 'Step L',
        joints: { ..._ZERO, l_hip_pitch: 25, l_knee: -15, l_ankle_pitch: 10, r_hip_pitch: -5 },
    },
    reach: {
        label: 'Reach',
        joints: { ..._ZERO, r_shoulder_pitch: 80, r_shoulder_roll: 15, r_elbow: 10,
                           l_shoulder_pitch: 80, l_shoulder_roll: -15, l_elbow: 10 },
    },
    wave: {
        label: 'Wave',
        joints: { ..._ZERO, r_shoulder_roll: 120, r_shoulder_pitch: 0, r_elbow: 40 },
    },
    t_pose: {
        label: 'T-Pose',
        joints: { ..._ZERO, r_shoulder_roll: 90, l_shoulder_roll: -90 },
    },
    look_up: {
        label: 'Look Up',
        joints: { ..._ZERO, neck_pitch: -30 },
    },
    look_right: {
        label: 'Look R',
        joints: { ..._ZERO, neck_yaw: 45 },
    },
    squat: {
        label: 'Squat',
        joints: { ..._ZERO, r_hip_pitch: 60, r_knee: -90, r_ankle_pitch: 30,
                           l_hip_pitch: 60, l_knee: -90, l_ankle_pitch: 30 },
    },
    twist: {
        label: 'Twist',
        joints: { ..._ZERO, waist_yaw: 45, r_shoulder_pitch: 30, l_shoulder_pitch: 30 },
    },
};

// Map joint names → motor channel for instant preview
const JOINT_CHANNEL = {
    // Waist
    waist_yaw: 'locomotion', waist_roll: 'locomotion', waist_pitch: 'locomotion',
    // Right leg
    r_hip_yaw: 'locomotion', r_hip_roll: 'locomotion', r_hip_pitch: 'locomotion',
    r_knee: 'locomotion', r_ankle_pitch: 'locomotion', r_ankle_roll: 'locomotion',
    // Left leg
    l_hip_yaw: 'locomotion', l_hip_roll: 'locomotion', l_hip_pitch: 'locomotion',
    l_knee: 'locomotion', l_ankle_pitch: 'locomotion', l_ankle_roll: 'locomotion',
    // Right arm
    r_shoulder_pitch: 'manipulation', r_shoulder_roll: 'manipulation',
    r_shoulder_yaw: 'manipulation', r_elbow: 'manipulation',
    r_wrist_pitch: 'manipulation', r_wrist_yaw: 'manipulation',
    // Left arm
    l_shoulder_pitch: 'manipulation', l_shoulder_roll: 'manipulation',
    l_shoulder_yaw: 'manipulation', l_elbow: 'manipulation',
    l_wrist_pitch: 'manipulation', l_wrist_yaw: 'manipulation',
    // Head
    neck_pitch: 'head', neck_yaw: 'head',
};

const PUSHES = [
    { label: 'Push Fwd',  body: 'torso', force: [0, 15, 0]  },
    { label: 'Push Back',  body: 'torso', force: [0, -15, 0] },
    { label: 'Push Right', body: 'torso', force: [15, 0, 0]  },
    { label: 'Push Left',  body: 'torso', force: [-15, 0, 0] },
    { label: 'Nudge Head', body: 'head',  force: [0, 5, 0]   },
];

export class BodyGuide {
    constructor(container, sendFn) {
        this._container = container;
        this._send = sendFn;  // (data) => void — sends via WS or REST
        this._joints = null;
        this._sliders = {};
        this.onPreview = null;  // (channel) => void — instant visual preview
        this._teachMode = false;
        this._teachRepeats = 3;
        this._teachBusy = false; // true while a teach sequence is running
        this._statusEl = null;   // teach progress indicator
    }

    async start(apiBase) {
        // Fetch joint info
        try {
            const resp = await fetch(apiBase + '/api/mujoco/joints');
            const json = await resp.json();
            this._joints = json.joints || [];
        } catch (e) {
            console.warn('Failed to fetch joint info:', e);
            this._joints = [];
        }
        this._render();
    }

    _render() {
        const el = this._container;
        el.innerHTML = '';

        // Section: Teach Mode toggle
        el.appendChild(this._makeSection('Teach Mode', () => {
            const wrap = document.createElement('div');
            wrap.className = 'guide-btn-row';
            wrap.style.alignItems = 'center';

            const toggle = document.createElement('button');
            toggle.className = 'guide-btn teach-toggle';
            toggle.textContent = 'Teach: OFF';
            toggle.onclick = () => {
                this._teachMode = !this._teachMode;
                toggle.textContent = this._teachMode ? 'Teach: ON' : 'Teach: OFF';
                toggle.classList.toggle('active', this._teachMode);
            };
            wrap.appendChild(toggle);

            const repLabel = document.createElement('span');
            repLabel.style.cssText = 'color:#aaa;font-size:12px;margin-left:8px;';
            repLabel.textContent = 'Reps:';
            wrap.appendChild(repLabel);

            const repSel = document.createElement('select');
            repSel.className = 'guide-select';
            for (const n of [1, 3, 5, 10]) {
                const opt = document.createElement('option');
                opt.value = n;
                opt.textContent = n;
                if (n === 3) opt.selected = true;
                repSel.appendChild(opt);
            }
            repSel.onchange = () => { this._teachRepeats = parseInt(repSel.value); };
            wrap.appendChild(repSel);

            // Status indicator
            const status = document.createElement('span');
            status.className = 'teach-status';
            status.style.cssText = 'color:#ccc;font-size:12px;margin-left:12px;';
            this._statusEl = status;
            wrap.appendChild(status);

            return wrap;
        }));

        // Section: Presets
        el.appendChild(this._makeSection('Guided Poses', () => {
            const row = document.createElement('div');
            row.className = 'guide-btn-row';
            for (const [key, preset] of Object.entries(PRESETS)) {
                const btn = document.createElement('button');
                btn.className = 'guide-btn preset';
                btn.textContent = preset.label;
                btn.onclick = () => this._sendPose(preset.joints);
                row.appendChild(btn);
            }
            return row;
        }));

        // Section: Push
        el.appendChild(this._makeSection('Push (Balance Training)', () => {
            const row = document.createElement('div');
            row.className = 'guide-btn-row';
            for (const push of PUSHES) {
                const btn = document.createElement('button');
                btn.className = 'guide-btn push';
                btn.textContent = push.label;
                btn.onclick = () => this._sendPush(push.body, push.force);
                row.appendChild(btn);
            }
            return row;
        }));

        // Section: Reward
        el.appendChild(this._makeSection('Reward Signal', () => {
            const row = document.createElement('div');
            row.className = 'guide-btn-row';
            for (const ch of ['locomotion', 'manipulation', 'head']) {
                const good = document.createElement('button');
                good.className = 'guide-btn reward-good';
                good.textContent = ch.slice(0, 4) + ' Good';
                good.onclick = () => this._sendReward(ch, true);
                row.appendChild(good);

                const bad = document.createElement('button');
                bad.className = 'guide-btn reward-bad';
                bad.textContent = ch.slice(0, 4) + ' Bad';
                bad.onclick = () => this._sendReward(ch, false);
                row.appendChild(bad);
            }
            return row;
        }));

        // Section: Joint Sliders — grouped by body region
        if (this._joints && this._joints.length > 0) {
            const groups = [
                { label: 'Waist',     filter: j => j.name.startsWith('waist') },
                { label: 'Head',      filter: j => j.name.startsWith('neck') },
                { label: 'R Arm',     filter: j => j.name.startsWith('r_shoulder') || j.name.startsWith('r_elbow') || j.name.startsWith('r_wrist') },
                { label: 'L Arm',     filter: j => j.name.startsWith('l_shoulder') || j.name.startsWith('l_elbow') || j.name.startsWith('l_wrist') },
                { label: 'R Leg',     filter: j => j.name.startsWith('r_hip') || j.name === 'r_knee' || j.name.startsWith('r_ankle') },
                { label: 'L Leg',     filter: j => j.name.startsWith('l_hip') || j.name === 'l_knee' || j.name.startsWith('l_ankle') },
            ];
            for (const grp of groups) {
                const joints = this._joints.filter(grp.filter);
                if (joints.length === 0) continue;
                el.appendChild(this._makeSection(grp.label, () => {
                    const wrap = document.createElement('div');
                    wrap.className = 'guide-sliders';
                    for (const j of joints) {
                        const row = document.createElement('div');
                        row.className = 'slider-row';

                        const label = document.createElement('span');
                        label.className = 'slider-label';
                        // Short display name: "r_shoulder_pitch" → "Sh Pitch"
                        label.textContent = this._shortName(j.name);

                        const input = document.createElement('input');
                        input.type = 'range';
                        input.className = 'slider-input';
                        input.min = j.range_deg[0];
                        input.max = j.range_deg[1];
                        input.value = 0;
                        input.step = 1;

                        const val = document.createElement('span');
                        val.className = 'slider-val';
                        val.textContent = '0';

                        input.oninput = () => {
                            val.textContent = input.value;
                        };
                        input.onchange = () => {
                            if (this._teachMode) {
                                this._sendPose(this._getAllSliderValues());
                            } else {
                                this._sendPose({ [j.name]: parseInt(input.value) });
                            }
                        };

                        this._sliders[j.name] = input;
                        row.appendChild(label);
                        row.appendChild(input);
                        row.appendChild(val);
                        wrap.appendChild(row);
                    }
                    return wrap;
                }));
            }
        }

        // Section: Reset
        const resetBtn = document.createElement('button');
        resetBtn.className = 'guide-btn reset-btn';
        resetBtn.textContent = 'Reset Body';
        resetBtn.onclick = () => this._sendReset();
        el.appendChild(resetBtn);
    }

    _shortName(name) {
        // "r_shoulder_pitch" → "Sh Pitch", "r_elbow" → "Elbow", "waist_yaw" → "Yaw"
        const parts = name.replace(/^[rl]_/, '').split('_');
        const abbr = {
            shoulder: 'Sh', hip: 'Hip', ankle: 'Ankle', wrist: 'Wrist',
            waist: 'Waist', neck: 'Neck', knee: 'Knee', elbow: 'Elbow',
        };
        return parts.map(p => abbr[p] || (p.charAt(0).toUpperCase() + p.slice(1))).join(' ');
    }

    _makeSection(title, contentFn) {
        const sec = document.createElement('div');
        sec.className = 'guide-section';
        const h = document.createElement('div');
        h.className = 'guide-section-title';
        h.textContent = title;
        sec.appendChild(h);
        sec.appendChild(contentFn());
        return sec;
    }

    _sendPose(joints) {
        // Instant client-side preview
        const ch = this._detectChannel(joints);
        if (ch && this.onPreview) this.onPreview(ch);

        if (this._teachMode) {
            // Teach mode: auto-pair with reward, repeat N times server-side
            this._send({ action: 'teach', joints, repeats: this._teachRepeats });
            this._setTeachStatus(`Teaching ${this._teachRepeats}x...`);
        } else {
            // Normal mode: single pose command
            this._send({ action: 'pose', joints });
        }
    }

    _setTeachStatus(msg) {
        if (this._statusEl) this._statusEl.textContent = msg;
    }

    /** Called externally when a teach progress message arrives from server. */
    updateTeachProgress(current, total) {
        if (current >= total) {
            this._setTeachStatus(`Done (${total}/${total})`);
            setTimeout(() => this._setTeachStatus(''), 3000);
        } else {
            this._setTeachStatus(`Teaching ${current}/${total}...`);
        }
    }

    _getAllSliderValues() {
        const joints = { ..._ZERO };
        for (const [name, slider] of Object.entries(this._sliders)) {
            joints[name] = parseInt(slider.value) || 0;
        }
        return joints;
    }

    _sendPush(body, force) {
        // Preview: push on torso → locomotion, head → head
        const ch = body === 'head' ? 'head' : 'locomotion';
        if (this.onPreview) this.onPreview(ch);
        this._send({ action: 'push', body, force });
    }

    _sendReward(channel, success) {
        this._send({ action: 'reward', channel, success });
    }

    _sendReset() {
        this._send({ action: 'reset' });
        // Reset sliders to 0
        for (const [, slider] of Object.entries(this._sliders)) {
            slider.value = 0;
            slider.nextElementSibling.textContent = '0';
        }
    }

    /**
     * Detect the primary motor channel from a joint angles dict.
     * Returns 'locomotion', 'manipulation', 'head', or null.
     */
    _detectChannel(joints) {
        const counts = {};
        for (const [name, angle] of Object.entries(joints)) {
            if (Math.abs(angle) < 0.5) continue;
            const ch = JOINT_CHANNEL[name];
            if (ch) counts[ch] = (counts[ch] || 0) + 1;
        }
        // Return channel with most active joints, or null
        let best = null, bestN = 0;
        for (const [ch, n] of Object.entries(counts)) {
            if (n > bestN) { best = ch; bestN = n; }
        }
        return best;
    }
}
