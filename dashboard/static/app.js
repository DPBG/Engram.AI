// ═══════════════════════════════════════════════════════════════════
// Engram Dashboard — Client
// ═══════════════════════════════════════════════════════════════════

class ActiveLearningAI {
    constructor() {
        this.ws = null;
        this.natsPaused = false;
        this.reconnectAttempts = 0;
        this.maxReconnect = 10;
        this.startTime = Date.now();
        this.chatBusy = false;
        this.lastModel = '—';
        this.gatewayData = null;
        this.gwLastSeen = 0;
        this.gwPendingSensors = new Set(); // sensor_ids with pending commands
        this.videoSessions = {}; // session_id -> status
        this.videoQueueState = null; // { queue, active, completed, queue_length }
        this.approvalTimers = {}; // trace_id -> timeoutId
        this.approvalsSent = new Set(); // trace_ids already responded to

        this.init();
    }

    init() {
        this.bindUI();
        this.connectWebSocket();
        this.fetchInitialData();
        this.startUptimeTimer();
        setInterval(() => this.fetchDockerMetrics(), 15000);
        setInterval(() => this.fetchSystemInfo(), 30000);
        setInterval(() => this.fetchFlywheel(), 20000);
        setInterval(() => this.fetchNeuro(), 5000);
        setInterval(() => this._checkGatewayStale(), 5000);
        setInterval(() => this.fetchVideoSessions(), 10000);
    }

    // ─── UI Bindings ─────────────────────────────────────────────────

    bindUI() {
        // ── Mobile drawer ──
        const hamburger = document.getElementById('hamburger');
        const drawer = document.getElementById('drawer');
        const overlay = document.getElementById('drawer-overlay');
        const drawerClose = document.getElementById('drawer-close');
        const openDrawer = () => {
            hamburger.classList.add('open');
            drawer.classList.add('open');
            overlay.classList.add('open');
            this._syncDrawerStats();
        };
        const closeDrawer = () => {
            hamburger.classList.remove('open');
            drawer.classList.remove('open');
            overlay.classList.remove('open');
        };
        if (hamburger) hamburger.addEventListener('click', () => {
            drawer.classList.contains('open') ? closeDrawer() : openDrawer();
        });
        if (overlay) overlay.addEventListener('click', closeDrawer);
        if (drawerClose) drawerClose.addEventListener('click', closeDrawer);

        const form = document.getElementById('chat-form');
        const input = document.getElementById('chat-input');
        form.addEventListener('submit', (e) => {
            e.preventDefault();
            const text = input.value.trim();
            if (!text || this.chatBusy) return;
            input.value = '';
            this.sendChat(text);
        });

        document.getElementById('nats-pause').addEventListener('click', (e) => {
            this.natsPaused = !this.natsPaused;
            e.currentTarget.classList.toggle('active', this.natsPaused);
            e.currentTarget.textContent = this.natsPaused ? '▶' : '⏸';
        });
        document.getElementById('nats-clear').addEventListener('click', () => {
            document.getElementById('nats-feed').innerHTML = '<div class="nats-empty">Cleared</div>';
        });

        document.getElementById('gw-start-all').addEventListener('click', (e) => {
            this.sendGatewayCommand({ action: 'start_all' }, e.currentTarget);
        });
        document.getElementById('gw-stop-all').addEventListener('click', (e) => {
            this.sendGatewayCommand({ action: 'stop_all' }, e.currentTarget);
        });

        // Video Training Queue
        document.getElementById('vt-submit').addEventListener('click', () => this.queueVideo());
        document.getElementById('vt-url').addEventListener('keydown', (e) => {
            if (e.key === 'Enter') { e.preventDefault(); this.queueVideo(); }
        });
        document.getElementById('vt-skip').addEventListener('click', () => this.skipVideo());
        document.getElementById('vt-clear-queue').addEventListener('click', () => this.clearQueue());
        document.getElementById('vt-blacklist').addEventListener('click', () => this.blacklistActive());
    }

    // ─── WebSocket ───────────────────────────────────────────────────

    connectWebSocket() {
        const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
        this.ws = new WebSocket(`${proto}//${location.host}/ws`);
        this.ws.onopen = () => { this.reconnectAttempts = 0; this.setConnection(true); };
        this.ws.onclose = () => { this.setConnection(false); this.scheduleReconnect(); };
        this.ws.onerror = () => this.setConnection(false);
        this.ws.onmessage = (e) => {
            try { this.handleWS(JSON.parse(e.data)); }
            catch (err) { console.error('WS parse:', err); }
        };
    }

    scheduleReconnect() {
        if (this.reconnectAttempts >= this.maxReconnect) return;
        this.reconnectAttempts++;
        setTimeout(() => this.connectWebSocket(), Math.min(2000 * this.reconnectAttempts, 15000));
    }

    setConnection(connected) {
        document.getElementById('conn-dot').className = 'conn-dot ' + (connected ? 'connected' : 'disconnected');
        document.getElementById('conn-text').textContent = connected ? 'Connected' : 'Reconnecting…';
    }

    handleWS(msg) {
        switch (msg.type) {
            case 'init':
                if (msg.data.system) this.renderSystemInfo(msg.data.system);
                if (msg.data.skills) this.renderSkills(msg.data.skills, msg.data.skills_by_category);
                if (msg.data.flywheel) this.renderFlywheel(msg.data.flywheel);
                if (msg.data.insights) this.renderInsights(msg.data.insights);
                if (msg.data.live_metrics) this.updateGauges(msg.data.live_metrics);
                if (msg.data.messages) msg.data.messages.forEach(m => this.appendNATSMessage(m));
                if (msg.data.neuromorphic) this.renderNeuro(msg.data.neuromorphic);
                if (msg.data.gateway) this.renderGateway(msg.data.gateway);
                if (msg.data.video_sessions) {
                    msg.data.video_sessions.forEach(s => { this.videoSessions[s.session_id] = s; });
                    this.renderVideoQueue();
                }
                this.updateTopStats(msg.data);
                break;
            case 'gateway_update':
                this.renderGateway(msg.data);
                break;
            case 'neuro_update':
                this.renderNeuro(msg.data);
                break;
            case 'metrics_update':
                if (msg.data.live) this.updateGauges(msg.data.live);
                if (msg.data.docker) this.renderDockerMetrics(msg.data.docker);
                break;
            case 'message':
                this.appendNATSMessage(msg.data);
                break;
            case 'insights':
                this.renderInsights(msg.data);
                break;
            case 'flywheel_update':
                this.renderFlywheel(msg.data);
                break;
            case 'skills_update':
                this.renderSkills(msg.data);
                break;
            case 'chat_response':
                this.removeChatTyping();
                this.lastModel = msg.data.model || '—';
                this.appendChatMessage('assistant', msg.data.reply, msg.data.model);
                document.getElementById('chat-model').textContent = this.lastModel;
                this.chatBusy = false;
                document.getElementById('chat-send').disabled = false;
                break;
            case 'video_training_update':
                if (msg.data && msg.data.type === 'queue_update') {
                    this.videoQueueState = msg.data;
                    this.renderVideoQueue();
                } else if (msg.data && msg.data.session_id &&
                    !['error', 'download_error', 'pending'].includes(msg.data.session_id)) {
                    this.videoSessions[msg.data.session_id] = msg.data;
                    this.renderVideoQueue();
                }
                break;
            case 'neuro_response':
                this.showNeuralReaction(msg.data);
                break;
            case 'approval_request':
                this.showApprovalRequest(msg.data);
                break;
            case 'mujoco_state':
                this.updateBodyStatus(msg.data);
                break;
            case 'visual_body_frame':
                this.updateBodySelfView(msg.data);
                break;
        }
    }

    updateBodyStatus(data) {
        const el = document.getElementById('body-status');
        if (!el) return;
        const h = data.torso_height != null ? data.torso_height.toFixed(2) : '?';
        const t = data.sim_time != null ? data.sim_time.toFixed(0) : '?';
        const ch = data.active_channel || 'idle';
        el.innerHTML = `<span style="color:var(--accent);">Live</span> &mdash; height ${h}m, ${ch}, sim ${t}s`;
    }

    updateBodySelfView(data) {
        if (!data || !data.pixels_b64) return;
        const canvas = document.getElementById('body-selfview-canvas');
        const placeholder = document.getElementById('body-selfview-placeholder');
        if (!canvas) return;
        // Show canvas, hide placeholder on first frame
        if (placeholder && placeholder.style.display !== 'none') {
            placeholder.style.display = 'none';
            canvas.style.display = 'block';
        }
        const w = data.width || 64, h = data.height || 64;
        const raw = atob(data.pixels_b64);
        const ctx = canvas.getContext('2d');
        const imgData = ctx.createImageData(w, h);
        for (let i = 0; i < raw.length; i++) {
            const v = raw.charCodeAt(i);
            imgData.data[i * 4]     = v;
            imgData.data[i * 4 + 1] = v;
            imgData.data[i * 4 + 2] = v;
            imgData.data[i * 4 + 3] = 255;
        }
        canvas.width = w;
        canvas.height = h;
        ctx.putImageData(imgData, 0, 0);
    }

    // ─── Fetch ───────────────────────────────────────────────────────

    async fetchInitialData() {
        await Promise.allSettled([
            this.fetchSystemInfo(),
            this.fetchDockerMetrics(),
            this.fetchSkills(),
            this.fetchFlywheel(),
            this.fetchInsights(),
            this.fetchNeuro(),
            this.fetchGateway(),
            this.fetchVideoSessions(),
        ]);
    }

    async fetchSystemInfo() {
        try {
            const d = await (await fetch('/api/system')).json();
            if (d.info) this.renderSystemInfo(d.info);
            if (d.live) this.updateGauges(d.live);
        } catch (e) { console.warn('system:', e); }
    }

    async fetchDockerMetrics() {
        try {
            const d = await (await fetch('/api/metrics')).json();
            if (d.metrics) this.renderDockerMetrics(d.metrics);
        } catch (e) { console.warn('metrics:', e); }
    }

    async fetchSkills() {
        try {
            const d = await (await fetch('/api/skills')).json();
            if (d.skills) this.renderSkills(d.skills, d.by_category);
            this.updateStatChip('stat-skills', d.skills?.length || 0);
            this.updateStatChip('stat-calls', d.total_calls || 0);
        } catch (e) { console.warn('skills:', e); }
    }

    async fetchFlywheel() {
        try {
            const d = await (await fetch('/api/flywheel')).json();
            this.renderFlywheel(d);
            this.updateStatChip('stat-knowledge', d.total_knowledge_entries || 0);
        } catch (e) { console.warn('flywheel:', e); }
    }

    async fetchInsights() {
        try {
            const d = await (await fetch('/api/insights')).json();
            if (d.insights) this.renderInsights(d.insights);
        } catch (e) { console.warn('insights:', e); }
    }

    async fetchNeuro() {
        try {
            const d = await (await fetch('/api/neuromorphic')).json();
            if (d.neuromorphic) this.renderNeuro(d.neuromorphic);
        } catch (e) { console.warn('neuro:', e); }
    }

    async fetchGateway() {
        try {
            const d = await (await fetch('/api/gateway')).json();
            if (d.gateway) this.renderGateway(d.gateway);
        } catch (e) { console.warn('gateway:', e); }
    }

    // ─── Render: Neuromorphic ─────────────────────────────────────────

    renderNeuro(data) {
        if (!data || !data.firing_rates) return;

        // Header stats
        const stepEl = document.getElementById('neuro-step');
        const totalEl = document.getElementById('neuro-total');
        if (stepEl) stepEl.textContent = this._formatNum(data.step_count || 0);
        if (totalEl) totalEl.textContent = this._formatNum(data.total_neurons || 0);

        // Firing rates
        const regionsEl = document.getElementById('neuro-regions');
        if (regionsEl && data.firing_rates) {
            const regionLabels = {
                brainstem: 'Brainstem',
                reflex_arc: 'Reflex Arc',
                sensory_cortex: 'Sensory',
                motor_cortex: 'Motor',
                cerebellum: 'Cerebellum',
                association_cortex: 'Association',
                predictive_layer: 'Predictive',
                working_memory: 'Working Mem',
            };
            let html = '';
            for (const [key, label] of Object.entries(regionLabels)) {
                const rate = data.firing_rates[key] || 0;
                const pct = Math.min(rate * 100, 100);
                let cls = '';
                if (pct > 30) cls = 'critical';
                else if (pct > 15) cls = 'hot';
                else if (pct > 3) cls = 'active';
                html += `
                    <div class="neuro-region-row">
                        <span class="neuro-region-name">${label}</span>
                        <div class="neuro-region-bar"><div class="neuro-region-fill ${cls}" style="width:${pct.toFixed(1)}%"></div></div>
                        <span class="neuro-region-val">${(rate * 100).toFixed(1)}%</span>
                    </div>`;
            }
            regionsEl.innerHTML = html;
        }

        // Drives
        if (data.drives) {
            this._setNeuroBar('energy', data.drives.energy || 0);
            this._setNeuroBar('damage', data.drives.damage || 0);
            this._setNeuroBar('temp', data.drives.temperature || 0);
            this._setNeuroBar('fatigue', data.drives.fatigue || 0);
        }

        // STDP Learning Rate Bars
        if (data.stdp_deltas) {
            const stdpEl = document.getElementById('neuro-stdp');
            if (stdpEl) {
                const labels = {
                    sensory_association: 'Sens→Assoc',
                    sensory_motor: 'Sens→Motor',
                    sensory_feature: 'Sens→Feat',
                    feature_association: 'Feat→Assoc',
                    association_concept: 'Assoc→Conc',
                    sensory_cerebellum: 'Sens→Cereb',
                    association_lateral: 'Assoc Lat',
                };
                const deltas = data.stdp_deltas;
                const vals = Object.values(deltas).filter(v => v > 0);
                const maxDelta = vals.length > 0 ? Math.max(...vals) : 0.001;

                let html = '';
                for (const [key, label] of Object.entries(labels)) {
                    const delta = deltas[key];
                    if (delta === undefined) continue;
                    const pct = Math.min((delta / Math.max(maxDelta, 1e-6)) * 100, 100);
                    let cls = '';
                    if (pct > 70) cls = 'hot';
                    else if (pct > 20) cls = 'active';
                    html += `
                        <div class="neuro-stdp-row">
                            <span class="neuro-stdp-name">${label}</span>
                            <div class="neuro-stdp-bar"><div class="neuro-stdp-fill ${cls}" style="width:${pct.toFixed(1)}%"></div></div>
                            <span class="neuro-stdp-val">${delta.toFixed(5)}</span>
                        </div>`;
                }
                stdpEl.innerHTML = html || '<div class="sys-loading">No STDP activity</div>';
            }
        }
    }

    _setNeuroBar(id, val) {
        const fill = document.getElementById(`neuro-drive-${id}`);
        const valEl = document.getElementById(`neuro-drive-${id}-val`);
        if (!fill || !valEl) return;
        const pct = Math.max(0, Math.min(100, val * 100));
        fill.style.width = `${pct}%`;
        valEl.textContent = val.toFixed(2);
    }

    _formatNum(n) {
        if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M';
        if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
        return String(n);
    }

    // ─── Render: Sensory Gateway ──────────────────────────────────────

    renderGateway(data) {
        if (!data || !data.sensors) return;
        this.gatewayData = data;
        this.gwLastSeen = Date.now();

        // Clear pending flags — server confirmed new state
        this.gwPendingSensors.clear();

        const statusEl = document.getElementById('gw-status');
        const sensorsEl = document.getElementById('gw-sensors');
        const controlsEl = document.getElementById('gw-controls');

        const isOnline = data.gateway === 'running';
        const runningCount = data.sensors.filter(s => s.running).length;
        statusEl.innerHTML = `<span class="gw-badge ${isOnline ? 'online' : 'offline'}">${isOnline ? 'Online' : 'Offline'}</span>`
            + (isOnline ? `<span class="gw-last-seen">${runningCount}/${data.sensors.length} active</span>` : '');

        // Show controls only when online
        controlsEl.style.display = isOnline ? 'flex' : 'none';

        const sensorIcons = {
            camera: '📷', mic: '🎤', voice: '🗣️', serial: '🔌',
            videofile: '🎬', audiofile: '🔊', transcript: '📝',
        };

        let html = '';
        for (const s of data.sensors) {
            const prefix = s.sensor_id.split('.')[0];
            const icon = sensorIcons[prefix] || '📡';
            const isPending = this.gwPendingSensors.has(s.sensor_id);
            const statusCls = isPending ? 'pending' : (s.running ? 'on' : 'off');
            const statusText = isPending ? '...' : (s.running ? 'ON' : 'OFF');

            // Extra metadata for video-related sensors
            let extraMeta = '';
            const isVideoType = ['videofile', 'audiofile', 'transcript'].includes(prefix);
            if (isVideoType && s.running) {
                const parts = [];
                if (s.frames_emitted != null) parts.push(`${this._formatNum(s.frames_emitted)} frames`);
                if (s.loop_count != null) parts.push(`loop ${s.loop_count}`);
                if (s.progress != null) parts.push(`${Math.round(s.progress * 100)}%`);
                if (parts.length > 0) extraMeta = `<div class="gw-sensor-meta gw-sensor-extra">${parts.join(' · ')}</div>`;
            }

            html += `
                <div class="gw-sensor-item">
                    <span class="gw-sensor-icon">${icon}</span>
                    <div class="gw-sensor-info">
                        <div class="gw-sensor-name">${this.esc(s.name)}</div>
                        <div class="gw-sensor-meta">${this.esc(s.sensor_id)} · ${s.hz} Hz</div>
                        ${extraMeta}
                    </div>
                    <span class="gw-sensor-status ${statusCls}">${statusText}</span>
                    <button class="gw-sensor-toggle ${s.running ? 'active' : ''} ${isPending ? 'pending' : ''}"
                            data-sensor-id="${this.esc(s.sensor_id)}"
                            data-running="${s.running}"
                            title="${s.running ? 'Stop' : 'Start'}">
                    </button>
                </div>`;
        }
        sensorsEl.innerHTML = html;

        // Bind toggle clicks
        sensorsEl.querySelectorAll('.gw-sensor-toggle').forEach(btn => {
            btn.addEventListener('click', () => {
                const sensorId = btn.dataset.sensorId;
                const running = btn.dataset.running === 'true';
                // Optimistic UI: immediately show pending state
                this.gwPendingSensors.add(sensorId);
                btn.classList.add('pending');
                btn.classList.toggle('active', !running); // flip visually
                const statusSpan = btn.parentElement.querySelector('.gw-sensor-status');
                if (statusSpan) {
                    statusSpan.className = 'gw-sensor-status pending';
                    statusSpan.textContent = '...';
                }
                this.sendGatewayCommand({
                    action: running ? 'stop' : 'start',
                    sensor_id: sensorId,
                });
            });
        });
    }

    _checkGatewayStale() {
        if (!this.gwLastSeen) return;
        const elapsed = Date.now() - this.gwLastSeen;
        if (elapsed > 10000 && this.gatewayData) {
            // No update in 10s — gateway probably disconnected
            const statusEl = document.getElementById('gw-status');
            const ago = Math.floor(elapsed / 1000);
            statusEl.innerHTML = `<span class="gw-badge stale">Stale</span><span class="gw-last-seen">Last seen ${ago}s ago</span>`;
        }
    }

    sendGatewayCommand(cmd, btnEl) {
        // Visual feedback on the button that was clicked
        if (btnEl) {
            btnEl.classList.add('gw-btn-sending');
            setTimeout(() => btnEl.classList.remove('gw-btn-sending'), 1500);
        }

        // Mark all sensors pending for bulk commands
        if ((cmd.action === 'start_all' || cmd.action === 'stop_all') && this.gatewayData) {
            for (const s of this.gatewayData.sensors) {
                this.gwPendingSensors.add(s.sensor_id);
            }
            // Optimistic UI for all toggles
            document.querySelectorAll('.gw-sensor-toggle').forEach(t => t.classList.add('pending'));
            document.querySelectorAll('.gw-sensor-status').forEach(s => {
                s.className = 'gw-sensor-status pending';
                s.textContent = '...';
            });
        }

        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ type: 'gateway_command', command: cmd }));
        } else {
            fetch('/api/gateway/command', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(cmd),
            }).catch(e => console.warn('gateway cmd:', e));
        }
    }

    // ─── Render: System Info ─────────────────────────────────────────

    renderSystemInfo(info) {
        const el = document.getElementById('sys-info');
        const rows = [];
        const os = info.os || {};
        rows.push(['OS', `${os.system || '?'} ${os.release || ''}`]);
        rows.push(['Arch', os.machine || '?']);
        rows.push(['Host', os.hostname || '?']);
        const cpu = info.cpu || {};
        if (cpu.model) rows.push(['CPU', cpu.model]);
        rows.push(['Cores', cpu.cores || '?']);
        const mem = info.memory || {};
        if (mem.total_gb) rows.push(['RAM', `${mem.total_gb} GB`]);
        const disk = info.disk || {};
        if (disk.total_gb) rows.push(['Disk', `${disk.free_gb} GB free / ${disk.total_gb} GB`]);
        const gpu = info.gpu;
        if (gpu && gpu.length) rows.push(['GPU', gpu[0].name || 'Detected']);

        let html = rows.map(([k, v]) => `
            <div class="sys-row">
                <span class="sys-label">${this.esc(k)}</span>
                <span class="sys-value" title="${this.esc(String(v))}">${this.esc(String(v))}</span>
            </div>
        `).join('');

        // Capabilities
        const caps = info.capabilities || [];
        if (caps.length) {
            html += '<div style="margin-top:6px; display:flex; flex-wrap:wrap; gap:2px;">';
            caps.forEach(c => {
                html += `<span class="sys-tag">${this.esc(c)}</span>`;
            });
            html += '</div>';
        }

        el.innerHTML = html;
    }

    // ─── Render: Gauges ──────────────────────────────────────────────

    updateGauges(live) {
        if (live.load_average) {
            const load = live.load_average['1min'] || 0;
            this.setGauge('cpu', Math.min(load * 25, 100), `${load.toFixed(1)}`);
        }
        if (live.memory) {
            const pct = live.memory.used_percent || 0;
            this.setGauge('mem', pct, `${pct.toFixed(0)}%`);
        }
        if (live.disk) {
            const pct = live.disk.used_percent || 0;
            this.setGauge('disk', pct, `${pct.toFixed(0)}%`);
        }
    }

    setGauge(id, pct, label) {
        const fill = document.getElementById(`gauge-${id}`);
        const val = document.getElementById(`gauge-${id}-val`);
        if (!fill || !val) return;
        pct = Math.max(0, Math.min(100, pct));
        fill.style.width = `${pct}%`;
        fill.className = 'bar-fill' + (pct > 85 ? ' crit' : pct > 65 ? ' warn' : '');
        val.textContent = label;
    }

    // ─── Render: Skills ──────────────────────────────────────────────

    renderSkills(skills, byCategory) {
        const el = document.getElementById('skill-list');

        // If we got categorized data, use it
        const cats = byCategory || this._categorizeSkills(skills);
        const catLabels = {
            perception: '👁️ Perception',
            cognition: '🧠 Cognition',
            memory: '💾 Memory',
            communication: '📡 Communication',
        };

        let html = '';
        for (const [cat, items] of Object.entries(cats)) {
            html += `<div class="skill-cat-label">${catLabels[cat] || cat}</div>`;
            for (const s of items) {
                html += `
                    <div class="skill-item" title="${this.esc(s.description || '')}">
                        <span class="skill-icon">${s.icon || '⚙️'}</span>
                        <div class="skill-info">
                            <div class="skill-name">${this.esc(s.name)}</div>
                            <div class="skill-meta">${this.esc(s.id)}${s.avg_ms ? ` · ~${s.avg_ms}ms` : ''}</div>
                        </div>
                        <span class="skill-calls">${s.calls || 0}</span>
                    </div>
                `;
            }
        }
        el.innerHTML = html;

        // Update top stats
        if (Array.isArray(skills)) {
            this.updateStatChip('stat-skills', skills.length);
            this.updateStatChip('stat-calls', skills.reduce((a, s) => a + (s.calls || 0), 0));
        }
    }

    _categorizeSkills(skills) {
        if (!Array.isArray(skills)) return {};
        const cats = {};
        skills.forEach(s => {
            const cat = s.category || 'other';
            if (!cats[cat]) cats[cat] = [];
            cats[cat].push(s);
        });
        return cats;
    }

    // ─── Render: Flywheel ────────────────────────────────────────────

    renderFlywheel(fw) {
        if (!fw) return;

        const sources = fw.sources || {};
        const teleop = sources.teleoperation || 0;
        const observe = sources.observation || 0;
        const deploy = sources.deployment || 0;
        const sim = sources.simulation || 0;
        const total = teleop + observe + deploy + sim || 1;

        // Update counts
        document.getElementById('fw-teleop').textContent = teleop;
        document.getElementById('fw-observe').textContent = observe;
        document.getElementById('fw-deploy').textContent = deploy;
        document.getElementById('fw-sim').textContent = sim;
        document.getElementById('fw-total').textContent = fw.total_knowledge_entries || 0;

        // Draw arcs
        const segments = [
            { id: 'fw-arc-teleop', val: teleop },
            { id: 'fw-arc-observe', val: observe },
            { id: 'fw-arc-deploy', val: deploy },
            { id: 'fw-arc-sim', val: sim },
        ];

        const cx = 100, cy = 100, r = 85;
        let startAngle = -90; // Start from top

        segments.forEach(seg => {
            const pct = seg.val / total;
            const sweep = pct * 360;
            const endAngle = startAngle + sweep;

            if (sweep > 0.5) {
                const path = this._arcPath(cx, cy, r, startAngle, endAngle - 2); // 2° gap
                document.getElementById(seg.id).setAttribute('d', path);
            } else {
                document.getElementById(seg.id).setAttribute('d', '');
            }

            startAngle = endAngle;
        });

        // Update header stat
        this.updateStatChip('stat-knowledge', fw.total_knowledge_entries || 0);
    }

    _arcPath(cx, cy, r, startDeg, endDeg) {
        const rad = d => d * Math.PI / 180;
        const x1 = cx + r * Math.cos(rad(startDeg));
        const y1 = cy + r * Math.sin(rad(startDeg));
        const x2 = cx + r * Math.cos(rad(endDeg));
        const y2 = cy + r * Math.sin(rad(endDeg));
        const large = (endDeg - startDeg) > 180 ? 1 : 0;
        return `M ${x1} ${y1} A ${r} ${r} 0 ${large} 1 ${x2} ${y2}`;
    }

    // ─── Render: Insights ────────────────────────────────────────────

    renderInsights(insights) {
        const el = document.getElementById('insight-list');
        const items = (Array.isArray(insights) ? insights : [insights]).slice(-12).reverse();
        el.innerHTML = items.map(i =>
            `<div class="insight-item ${i.level || 'info'}">${this.esc(i.message)}</div>`
        ).join('');
    }

    // ─── Render: Docker ──────────────────────────────────────────────

    renderDockerMetrics(metrics) {
        const el = document.getElementById('container-list');
        if (!metrics || !metrics.length) {
            el.innerHTML = '<div class="sys-loading">No containers</div>';
            return;
        }
        el.innerHTML = metrics.map(m => `
            <div class="ctn-item">
                <div class="ctn-name">${this.esc(m.service)}</div>
                <div class="ctn-row"><span>CPU</span><span class="val">${m.cpu_percent.toFixed(1)}%</span></div>
                <div class="ctn-row"><span>Mem</span><span class="val">${m.memory_mb.toFixed(0)} MB</span></div>
            </div>
        `).join('');
    }

    // ─── Top Stats ───────────────────────────────────────────────────

    updateTopStats(data) {
        if (data.skills) this.updateStatChip('stat-skills', Array.isArray(data.skills) ? data.skills.length : 0);
        if (data.flywheel) this.updateStatChip('stat-knowledge', data.flywheel.total_knowledge_entries || 0);
    }

    updateStatChip(id, val) {
        const el = document.getElementById(id);
        if (el) el.textContent = val;
    }

    // ─── Chat ────────────────────────────────────────────────────────

    async sendChat(text) {
        this.chatBusy = true;
        document.getElementById('chat-send').disabled = true;
        this.appendChatMessage('user', text);
        this.showChatTyping();

        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ type: 'chat', message: text }));
        } else {
            try {
                const d = await (await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: text }),
                })).json();
                this.removeChatTyping();
                this.lastModel = d.model || '—';
                this.appendChatMessage('assistant', d.reply, d.model);
                document.getElementById('chat-model').textContent = this.lastModel;
            } catch (err) {
                this.removeChatTyping();
                this.appendChatMessage('assistant', `⚠️ Error: ${err.message}`);
            }
            this.chatBusy = false;
            document.getElementById('chat-send').disabled = false;
        }

        // Refresh flywheel after chat (new teleoperation data)
        setTimeout(() => this.fetchFlywheel(), 500);
    }

    appendChatMessage(role, content, model) {
        const container = document.getElementById('chat-messages');
        const div = document.createElement('div');
        div.className = `msg ${role}`;
        const avatar = role === 'user' ? '👤' : '🧠';
        const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        let sourceTag = '';
        if (role === 'assistant' && model) {
            if (model === 'neural-only') {
                sourceTag = '<span class="msg-source msg-source-brain">brain</span>';
            } else if (model.startsWith('ollama/')) {
                sourceTag = `<span class="msg-source msg-source-llm">${model.replace('ollama/', '')}</span>`;
            } else if (model !== 'error') {
                sourceTag = `<span class="msg-source msg-source-llm">${model}</span>`;
            }
        }
        div.innerHTML = `
            <div class="msg-avatar">${avatar}</div>
            <div class="msg-body">
                <div class="msg-content">${this.renderMarkdown(content)}</div>
                <div class="msg-meta"><span class="msg-time">${time}</span>${sourceTag}</div>
            </div>
        `;
        container.appendChild(div);
        container.scrollTop = container.scrollHeight;
    }

    showChatTyping() {
        const c = document.getElementById('chat-messages');
        const div = document.createElement('div');
        div.className = 'msg assistant'; div.id = 'chat-typing';
        div.innerHTML = `<div class="msg-avatar">🧠</div><div class="msg-body"><div class="msg-content msg-typing"><span></span><span></span><span></span></div></div>`;
        c.appendChild(div); c.scrollTop = c.scrollHeight;
    }

    removeChatTyping() { const el = document.getElementById('chat-typing'); if (el) el.remove(); }

    showApprovalRequest(data) {
        const traceId = data.trace_id || '?';

        // Deduplicate — skip if already showing this approval
        if (this.approvalsSent.has(traceId) || this.approvalTimers[traceId]) {
            return;
        }

        const channel = data.channel || 'unknown';
        const intensity = data.intensity != null ? (data.intensity * 100).toFixed(0) : '?';
        const reason = data.reason || 'Requires human approval';
        const risk = data.risk_score != null ? (data.risk_score * 100).toFixed(0) : '?';
        const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

        const container = document.getElementById('chat-messages');
        const div = document.createElement('div');
        div.className = 'msg approval';
        div.id = `approval-${traceId}`;

        // Build content safely using textContent to avoid XSS
        const avatar = document.createElement('div');
        avatar.className = 'msg-avatar';
        avatar.textContent = '\u26A0';

        const body = document.createElement('div');
        body.className = 'msg-body';

        const content = document.createElement('div');
        content.className = 'msg-content';

        const header = document.createElement('div');
        header.className = 'approval-header';
        header.textContent = 'Safety Gate: DEFER';

        const detail1 = document.createElement('div');
        detail1.className = 'approval-detail';
        const strong = document.createElement('strong');
        strong.textContent = channel;
        detail1.appendChild(strong);
        detail1.appendChild(document.createTextNode(` at ${intensity}% intensity`));

        const detail2 = document.createElement('div');
        detail2.className = 'approval-detail';
        detail2.textContent = `Risk: ${risk}% \u2014 ${reason}`;

        const actions = document.createElement('div');
        actions.className = 'approval-actions';
        actions.id = `approval-actions-${traceId}`;

        const btnAllow = document.createElement('button');
        btnAllow.className = 'btn-approve';
        btnAllow.textContent = 'Allow';
        btnAllow.addEventListener('click', () => this.sendApprovalResponse(traceId, channel, true));

        const btnDeny = document.createElement('button');
        btnDeny.className = 'btn-deny';
        btnDeny.textContent = 'Deny';
        btnDeny.addEventListener('click', () => this.sendApprovalResponse(traceId, channel, false));

        actions.appendChild(btnAllow);
        actions.appendChild(btnDeny);

        const timeEl = document.createElement('div');
        timeEl.className = 'msg-time';
        timeEl.textContent = time;

        content.appendChild(header);
        content.appendChild(detail1);
        content.appendChild(detail2);
        content.appendChild(actions);
        body.appendChild(content);
        body.appendChild(timeEl);
        div.appendChild(avatar);
        div.appendChild(body);

        container.appendChild(div);
        container.scrollTop = container.scrollHeight;

        // Auto-expire after 30 seconds (fail-open: allow by default)
        this.approvalTimers[traceId] = setTimeout(() => {
            if (!this.approvalsSent.has(traceId)) {
                this.sendApprovalResponse(traceId, channel, true);
            }
        }, 30000);
    }

    sendApprovalResponse(traceId, channel, approved) {
        // Prevent double-send (race between user click and auto-expire timer)
        if (this.approvalsSent.has(traceId)) return;
        this.approvalsSent.add(traceId);

        // Clear auto-expire timer
        if (this.approvalTimers[traceId]) {
            clearTimeout(this.approvalTimers[traceId]);
            delete this.approvalTimers[traceId];
        }

        // Update UI
        const actions = document.getElementById(`approval-actions-${traceId}`);
        if (actions) {
            const result = document.createElement('span');
            result.className = `approval-result ${approved ? 'approved' : 'denied'}`;
            result.textContent = approved ? 'Allowed' : 'Denied';
            actions.replaceChildren(result);
        }

        // Send to backend
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({
                type: 'approval_response',
                data: { trace_id: traceId, channel: channel, approved: approved },
            }));
        }
    }

    showNeuralReaction(data) {
        const action = data.action || {};
        const channel = action.channel || action.type || 'unknown';
        const intensity = (action.intensity != null) ? (action.intensity * 100).toFixed(0) : '?';
        const step = (data.metadata || {}).step || '?';
        const source = data.provenance || '';

        let label;
        if (source === 'neuromorphic.reflex') {
            label = `reflex: ${action.reflex || channel}`;
        } else {
            label = channel;
        }

        const container = document.getElementById('neural-toasts');
        if (!container) return;
        const toast = document.createElement('div');
        toast.className = 'neural-toast';
        toast.innerHTML = `
            <span class="toast-icon">&#9889;</span>
            <span class="toast-label">${this.esc(label)}</span>
            <span class="toast-intensity">${this.esc(String(intensity))}%</span>
            <span class="toast-step">step ${this.esc(String(step))}</span>
        `;
        container.appendChild(toast);
        // Remove after animation completes (4s total)
        setTimeout(() => toast.remove(), 4000);
        // Cap visible toasts
        while (container.children.length > 3) container.firstChild.remove();
    }

    // ─── Video Training ────────────────────────────────────────────────

    async fetchVideoSessions() {
        try {
            const d = await (await fetch('/api/video/sessions')).json();
            if (d.sessions) {
                d.sessions.forEach(s => { this.videoSessions[s.session_id] = s; });
                this.renderVideoQueue();
            }
        } catch (e) { console.warn('video sessions:', e); }
    }

    queueVideo() {
        const urlEl = document.getElementById('vt-url');
        const url = urlEl.value.trim();
        if (!url) return;

        const fps = parseFloat(document.getElementById('vt-fps').value) || 2;
        const transcript = document.getElementById('vt-transcript').checked;
        const targetLoops = parseInt(document.getElementById('vt-loops').value) || 5;
        const category = document.getElementById('vt-category').value || '';

        const btn = document.getElementById('vt-submit');
        btn.disabled = true;
        btn.textContent = 'Adding...';
        setTimeout(() => { btn.disabled = false; btn.textContent = '+ Add to Queue'; }, 2000);

        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({
                type: 'video_queue',
                url, fps, transcript, category,
                target_loops: targetLoops,
            }));
        } else {
            fetch('/api/video/queue', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ url, fps, transcript, category, target_loops: targetLoops }),
            }).catch(e => console.warn('video queue:', e));
        }

        urlEl.value = '';
    }

    skipVideo() {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ type: 'video_skip' }));
        } else {
            fetch('/api/video/skip', { method: 'POST' }).catch(e => console.warn('skip:', e));
        }
    }

    clearQueue() {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ type: 'video_clear_queue' }));
        } else {
            fetch('/api/video/clear-queue', { method: 'POST' }).catch(e => console.warn('clear:', e));
        }
    }

    blacklistActive() {
        if (!this.videoQueueState || !this.videoQueueState.active) return;
        const sessionId = this.videoQueueState.active.session_id;
        const title = this.videoQueueState.active.title || 'video';
        const reason = prompt(`Why blacklist "${title}"? (e.g. bad quality, wrong content)`, 'Bad training data');
        if (!reason) return;

        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({
                type: 'video_blacklist',
                session_id: sessionId,
                reason: reason,
            }));
        } else {
            fetch('/api/video/blacklist', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: sessionId, reason }),
            }).catch(e => console.warn('blacklist:', e));
        }
    }

    removeQueued(sessionId) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ type: 'video_remove_queued', session_id: sessionId }));
        } else {
            fetch('/api/video/remove-queued', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: sessionId }),
            }).catch(e => console.warn('remove queued:', e));
        }
    }

    stopVideoSession(sessionId) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ type: 'video_stop', session_id: sessionId }));
        } else {
            fetch('/api/video/stop', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: sessionId }),
            }).catch(e => console.warn('video stop:', e));
        }
    }

    renderVideoQueue() {
        const npEl = document.getElementById('vt-now-playing');
        const queueHeaderEl = document.getElementById('vt-queue-header');
        const queueEl = document.getElementById('vt-queue');
        const completedHeaderEl = document.getElementById('vt-completed-header');
        const completedEl = document.getElementById('vt-completed');
        const summaryEl = document.getElementById('vt-summary');

        const qs = this.videoQueueState;
        if (!qs) {
            npEl.style.display = 'none';
            queueHeaderEl.style.display = 'none';
            queueEl.innerHTML = '';
            completedHeaderEl.style.display = 'none';
            completedEl.innerHTML = '';
            summaryEl.innerHTML = '';
            return;
        }

        // Now Playing
        if (qs.active) {
            const a = qs.active;
            const pct = Math.round((a.progress || 0) * 100);
            const loopTarget = a.target_loops > 0 ? a.target_loops : '∞';
            const catLabel = a.category ? ` [${a.category}]` : '';
            npEl.style.display = '';
            document.getElementById('vt-np-title').textContent = (a.title || 'Video') + catLabel;
            document.getElementById('vt-np-meta').innerHTML =
                `Loop ${a.loop_count || 0}/${loopTarget} · ${this._formatDuration(a.elapsed_s || 0)} · ${a.fps || 2} FPS · ${this._formatNum(a.frames_emitted || 0)} frames${a.transcript ? ' · STT' : ''}`;
            document.getElementById('vt-np-progress').style.width = pct + '%';
            document.getElementById('vt-np-progress-label').textContent =
                `${pct}% through current loop`;

            // Learning metrics display
            const lmEl = document.getElementById('vt-learning-metrics');
            if (lmEl && (a.learning_score !== undefined)) {
                lmEl.style.display = '';
                document.getElementById('vt-lm-score').textContent = (a.learning_score || 0).toFixed(4);
                document.getElementById('vt-lm-rate').textContent = (a.avg_learning_rate || 0).toFixed(6);
            } else if (lmEl) {
                lmEl.style.display = 'none';
            }

            // Convergence display
            const convEl = document.getElementById('vt-convergence');
            if (a.convergence && convEl) {
                const c = a.convergence;
                convEl.style.display = '';
                const badge = document.getElementById('vt-conv-badge');
                const convBar = document.getElementById('vt-conv-progress');
                const convDelta = document.getElementById('vt-conv-delta');
                const isConverged = c.converged;
                const stablePct = Math.min((c.stable_count / Math.max(c.window, 1)) * 100, 100);

                badge.textContent = isConverged ? 'Converged' : 'Adapting';
                badge.className = 'vt-conv-badge ' + (isConverged ? 'converged' : 'adapting');
                convBar.style.width = stablePct + '%';
                convBar.className = 'vt-conv-bar-fill' + (isConverged ? ' converged' : '');
                convDelta.textContent = `delta: ${c.mean_delta.toFixed(6)} · ${c.stable_count}/${c.window} stable`;
            } else if (convEl) {
                convEl.style.display = 'none';
            }
        } else {
            npEl.style.display = 'none';
        }

        // Queue
        const queueItems = qs.queue || [];
        if (queueItems.length > 0) {
            queueHeaderEl.style.display = '';
            document.getElementById('vt-queue-count').textContent = queueItems.length;
            let html = '';
            for (let i = 0; i < queueItems.length; i++) {
                const s = queueItems[i];
                const loopTarget = s.target_loops > 0 ? s.target_loops : '∞';
                const statusLabel = s.status === 'downloading' ? '(downloading...)' : '';
                html += `
                    <div class="vt-queue-item">
                        <span class="vt-qi-pos">${i + 1}</span>
                        <span class="vt-qi-title">${this.esc(s.title || 'Video')} ${statusLabel}</span>
                        <span class="vt-qi-loops">${loopTarget}x</span>
                        <button class="vt-qi-remove" data-session="${this.esc(s.session_id)}" title="Remove">✕</button>
                    </div>`;
            }
            queueEl.innerHTML = html;
            queueEl.querySelectorAll('.vt-qi-remove').forEach(btn => {
                btn.addEventListener('click', () => this.removeQueued(btn.dataset.session));
            });
        } else {
            queueHeaderEl.style.display = 'none';
            queueEl.innerHTML = '';
        }

        // Completed
        const completed = qs.completed || [];
        if (completed.length > 0) {
            completedHeaderEl.style.display = '';
            let html = '';
            for (const s of completed.slice(0, 10)) {
                const loopTarget = s.target_loops > 0 ? s.target_loops : '∞';
                const statusIcon = s.status === 'completed' ? '✓' : (s.status === 'error' ? '✗' : '—');
                const statusCls = s.status === 'completed' ? 'done' : (s.status === 'error' ? 'error' : '');
                const lsLabel = s.learning_score > 0 ? ` · score: ${s.learning_score.toFixed(3)}` : '';
                const catLabel = s.category ? ` [${s.category}]` : '';
                html += `
                    <div class="vt-completed-item ${statusCls}">
                        <span class="vt-ci-icon">${statusIcon}</span>
                        <span class="vt-ci-title">${this.esc(s.title || 'Video')}${catLabel}</span>
                        <span class="vt-ci-meta">${s.loop_count || 0}/${loopTarget} loops · ${this._formatNum(s.frames_emitted || 0)} frames${lsLabel}</span>
                    </div>`;
            }
            completedEl.innerHTML = html;
        } else {
            completedHeaderEl.style.display = 'none';
            completedEl.innerHTML = '';
        }

        // Summary
        const totalLoops = (completed.reduce((a, s) => a + (s.loop_count || 0), 0)) +
            (qs.active ? (qs.active.loop_count || 0) : 0);
        const totalFrames = (completed.reduce((a, s) => a + (s.frames_emitted || 0), 0)) +
            (qs.active ? (qs.active.frames_emitted || 0) : 0);
        const isActive = !!qs.active;
        if (isActive || totalLoops > 0) {
            summaryEl.innerHTML = `<span>${isActive ? '1 training' : 'idle'}</span><span>${queueItems.length} queued</span><span>${totalLoops} loops</span><span>${this._formatNum(totalFrames)} frames</span>`;
        } else {
            summaryEl.innerHTML = '';
        }
    }

    _formatDuration(seconds) {
        const s = Math.floor(seconds);
        const h = Math.floor(s / 3600);
        const m = Math.floor((s % 3600) / 60);
        const sec = s % 60;
        if (h > 0) return `${h}h ${String(m).padStart(2, '0')}m`;
        return `${m}m ${String(sec).padStart(2, '0')}s`;
    }

    // ─── NATS Feed ───────────────────────────────────────────────────

    appendNATSMessage(msg) {
        if (this.natsPaused) return;
        const feed = document.getElementById('nats-feed');
        const empty = feed.querySelector('.nats-empty');
        if (empty) empty.remove();

        const div = document.createElement('div');
        div.className = 'nats-msg';
        const time = msg.timestamp ? new Date(msg.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }) : '';
        const data = typeof msg.data === 'string' ? msg.data : JSON.stringify(msg.data, null, 2);
        const droppedBadge = msg.dropped ? `<span class="nats-dropped">+${msg.dropped} skipped</span>` : '';
        div.innerHTML = `
            <div class="nats-msg-head">
                <span class="nats-subject">${this.esc(msg.subject || '?')}</span>
                ${droppedBadge}
                <span class="nats-time">${time}</span>
            </div>
            <div class="nats-data">${this.esc(data.substring(0, 200))}</div>
        `;
        const dataEl = div.querySelector('.nats-data');
        dataEl.addEventListener('click', () => {
            dataEl.classList.toggle('expanded');
            dataEl.textContent = dataEl.classList.contains('expanded') ? data : this.esc(data.substring(0, 200));
        });
        feed.insertBefore(div, feed.firstChild);
        while (feed.children.length > 80) feed.lastChild.remove();
    }

    // ─── Uptime ──────────────────────────────────────────────────────

    startUptimeTimer() {
        // Fetch server uptime so we show actual server uptime, not page-load age
        fetch('/api/health').then(r => r.json()).then(d => {
            if (d.uptime_seconds) this.startTime = Date.now() - d.uptime_seconds * 1000;
        }).catch(() => {});
        setInterval(() => {
            const s = Math.floor((Date.now() - this.startTime) / 1000);
            const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
            document.getElementById('uptime').textContent = `${h}h ${String(m).padStart(2,'0')}m ${String(sec).padStart(2,'0')}s`;
        }, 1000);
    }

    // ─── Helpers ─────────────────────────────────────────────────────

    esc(text) { const el = document.createElement('span'); el.textContent = text; return el.innerHTML; }

    _syncDrawerStats() {
        // Mirror stats to drawer
        ['skills', 'calls', 'knowledge'].forEach(k => {
            const src = document.getElementById(`stat-${k}`);
            const dst = document.getElementById(`m-stat-${k}`);
            if (src && dst) dst.textContent = src.textContent;
        });
        // Mirror gauges
        ['cpu', 'mem', 'disk'].forEach(k => {
            const srcFill = document.getElementById(`gauge-${k}`);
            const srcVal = document.getElementById(`gauge-${k}-val`);
            const dstFill = document.getElementById(`m-gauge-${k}`);
            const dstVal = document.getElementById(`m-gauge-${k}-val`);
            if (srcFill && dstFill) dstFill.style.width = srcFill.style.width;
            if (srcVal && dstVal) dstVal.textContent = srcVal.textContent;
        });
        // Mirror connection
        const srcDot = document.getElementById('conn-dot');
        const dstDot = document.getElementById('m-conn-dot');
        if (srcDot && dstDot) dstDot.className = srcDot.className;
        const srcConn = document.getElementById('conn-text');
        const dstConn = document.getElementById('m-conn-text');
        if (srcConn && dstConn) dstConn.textContent = srcConn.textContent;
    }

    renderMarkdown(text) {
        if (!text) return '';
        let h = this.esc(text);
        h = h.replace(/```(\w*)\n?([\s\S]*?)```/g, '<pre><code>$2</code></pre>');
        h = h.replace(/`([^`]+)`/g, '<code>$1</code>');
        h = h.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
        h = h.replace(/\*(.+?)\*/g, '<em>$1</em>');
        h = h.replace(/\n/g, '<br>');
        return h;
    }
}

// ─── Boot ─────────────────────────────────────────────────────────
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => new ActiveLearningAI());
} else {
    new ActiveLearningAI();
}
