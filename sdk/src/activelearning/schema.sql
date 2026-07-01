-- ActiveLearningAI Unified Database Schema
-- SQLite schema for all services
-- Created: 2026-02-03

-- =============================================================================
-- MEMORY TABLES
-- =============================================================================

CREATE TABLE IF NOT EXISTS memory_episodes (
    id TEXT PRIMARY KEY,
    trace_id TEXT NOT NULL,
    timestamp INTEGER NOT NULL,
    embedding_ref TEXT,  -- Reference to Qdrant vector ID
    tags TEXT,  -- JSON array of tags
    utility_score REAL DEFAULT 0.0,
    episode_data TEXT,  -- JSON serialized episode data
    created_at INTEGER NOT NULL,
    UNIQUE(trace_id)
);

CREATE INDEX IF NOT EXISTS idx_memory_episodes_timestamp
    ON memory_episodes(timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_memory_episodes_utility
    ON memory_episodes(utility_score DESC);

CREATE INDEX IF NOT EXISTS idx_memory_episodes_trace_id
    ON memory_episodes(trace_id);

-- =============================================================================
-- BELIEF GRAPH TABLES
-- =============================================================================

CREATE TABLE IF NOT EXISTS belief_nodes (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL CHECK(type IN ('value', 'norm', 'fact')),
    content TEXT NOT NULL,
    confidence REAL NOT NULL CHECK(confidence >= 0.0 AND confidence <= 1.0),
    source TEXT NOT NULL DEFAULT 'unknown',
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    metadata TEXT  -- JSON serialized metadata
);

CREATE INDEX IF NOT EXISTS idx_belief_nodes_type
    ON belief_nodes(type);

CREATE INDEX IF NOT EXISTS idx_belief_nodes_confidence
    ON belief_nodes(confidence DESC);

CREATE TABLE IF NOT EXISTS belief_edges (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL CHECK(type IN ('supports', 'contradicts', 'entails', 'refines')),
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    strength REAL NOT NULL CHECK(strength >= 0.0 AND strength <= 1.0),
    evidence TEXT,
    created_at INTEGER NOT NULL,
    FOREIGN KEY (source_id) REFERENCES belief_nodes(id) ON DELETE CASCADE,
    FOREIGN KEY (target_id) REFERENCES belief_nodes(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_belief_edges_source
    ON belief_edges(source_id);

CREATE INDEX IF NOT EXISTS idx_belief_edges_target
    ON belief_edges(target_id);

CREATE INDEX IF NOT EXISTS idx_belief_edges_type
    ON belief_edges(type);

-- =============================================================================
-- KERNEL TABLES
-- =============================================================================

CREATE TABLE IF NOT EXISTS kernel_decisions (
    id TEXT PRIMARY KEY,
    trace_id TEXT NOT NULL,
    proposal_type TEXT NOT NULL CHECK(proposal_type IN ('action', 'code', 'cognitive', 'speech')),
    decision_type TEXT NOT NULL CHECK(decision_type IN ('ALLOW', 'TRANSFORM', 'DENY', 'DEFER')),
    source TEXT,                          -- 'neuromorphic' | 'planner' | 'meta-programmer'
    reason TEXT,
    risk_score REAL NOT NULL DEFAULT 0.0,
    flags TEXT,                           -- JSON array of risk flags
    norm_violations TEXT,                 -- JSON array of violated norms
    body_profile TEXT,                    -- active body profile name
    issued_at INTEGER NOT NULL,
    expires_at INTEGER,
    created_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now') * 1000)
);

CREATE INDEX IF NOT EXISTS idx_kernel_decisions_trace_id
    ON kernel_decisions(trace_id);

CREATE INDEX IF NOT EXISTS idx_kernel_decisions_type
    ON kernel_decisions(decision_type);

CREATE INDEX IF NOT EXISTS idx_kernel_decisions_timestamp
    ON kernel_decisions(issued_at DESC);

-- =============================================================================
-- LLM CACHE TABLES
-- =============================================================================

CREATE TABLE IF NOT EXISTS llm_cache (
    id TEXT PRIMARY KEY,
    prompt_hash TEXT NOT NULL UNIQUE,
    prompt TEXT NOT NULL,
    response TEXT NOT NULL,
    embedding_ref TEXT,  -- Reference to Qdrant vector ID
    tags TEXT NOT NULL DEFAULT '[]',  -- JSON array of invalidation tags
    confidence REAL DEFAULT 1.0,
    hit_count INTEGER DEFAULT 0,
    last_hit_at INTEGER,
    created_at INTEGER NOT NULL,
    expires_at INTEGER
);

CREATE INDEX IF NOT EXISTS idx_llm_cache_prompt_hash
    ON llm_cache(prompt_hash);

CREATE INDEX IF NOT EXISTS idx_llm_cache_hit_count
    ON llm_cache(hit_count DESC);

CREATE INDEX IF NOT EXISTS idx_llm_cache_last_hit
    ON llm_cache(last_hit_at DESC);

-- =============================================================================
-- OVERRIDE TABLES
-- =============================================================================

CREATE TABLE IF NOT EXISTS human_overrides (
    id TEXT PRIMARY KEY,
    trace_id TEXT NOT NULL,
    override_type TEXT NOT NULL CHECK(override_type IN ('operational', 'knowledge')),
    prompt TEXT NOT NULL,
    parameter_path TEXT,
    value TEXT,  -- JSON serialized value
    embedding_ref TEXT,  -- Reference to Qdrant vector ID for semantic lookup
    verification_method TEXT CHECK(verification_method IN ('camera', 'microphone', 'button', 'none')),
    verification_confidence REAL,
    applied_at INTEGER,
    created_at INTEGER NOT NULL,
    metadata TEXT  -- JSON serialized metadata
);

CREATE INDEX IF NOT EXISTS idx_human_overrides_trace_id
    ON human_overrides(trace_id);

CREATE INDEX IF NOT EXISTS idx_human_overrides_type
    ON human_overrides(override_type);

CREATE INDEX IF NOT EXISTS idx_human_overrides_applied_at
    ON human_overrides(applied_at DESC);

-- =============================================================================
-- EXTERNAL API TABLES
-- =============================================================================

CREATE TABLE IF NOT EXISTS external_api_queries (
    id TEXT PRIMARY KEY,
    trace_id TEXT NOT NULL,
    query TEXT NOT NULL,
    provider TEXT NOT NULL CHECK(provider IN ('claude', 'openai', 'other')),
    response TEXT,
    local_knowledge TEXT,  -- Local response before external query
    conflict_detected BOOLEAN DEFAULT FALSE,
    resolution TEXT,  -- How conflict was resolved
    kernel_approved BOOLEAN DEFAULT FALSE,
    created_at INTEGER NOT NULL,
    metadata TEXT  -- JSON serialized metadata
);

CREATE INDEX IF NOT EXISTS idx_external_api_queries_trace_id
    ON external_api_queries(trace_id);

CREATE INDEX IF NOT EXISTS idx_external_api_queries_provider
    ON external_api_queries(provider);

CREATE INDEX IF NOT EXISTS idx_external_api_queries_conflict
    ON external_api_queries(conflict_detected);

-- =============================================================================
-- TASK STORAGE TABLES
-- =============================================================================

CREATE TABLE IF NOT EXISTS learned_tasks (
    id TEXT PRIMARY KEY,
    task_name TEXT NOT NULL,
    task_description TEXT NOT NULL,
    file_path TEXT NOT NULL,  -- Path to task.py
    embedding_ref TEXT,  -- Reference to Qdrant vector ID
    success_rate REAL DEFAULT 0.0,
    execution_count INTEGER DEFAULT 0,
    avg_execution_time_ms REAL,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    metadata TEXT  -- JSON serialized metadata (sensors used, parameters, etc.)
);

CREATE INDEX IF NOT EXISTS idx_learned_tasks_success_rate
    ON learned_tasks(success_rate DESC);

CREATE INDEX IF NOT EXISTS idx_learned_tasks_execution_count
    ON learned_tasks(execution_count DESC);

CREATE INDEX IF NOT EXISTS idx_learned_tasks_created_at
    ON learned_tasks(created_at DESC);

-- =============================================================================
-- AUDIT TRAIL TABLE (Unified)
-- =============================================================================

CREATE TABLE IF NOT EXISTS audit_entries (
    id TEXT PRIMARY KEY,
    trace_id TEXT NOT NULL,
    timestamp INTEGER NOT NULL,
    component TEXT NOT NULL,
    action TEXT NOT NULL,
    details TEXT,  -- JSON serialized details
    code_hash TEXT,  -- SHA-256 hash of code if applicable
    created_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now') * 1000)
);

CREATE INDEX IF NOT EXISTS idx_audit_entries_trace_id
    ON audit_entries(trace_id);

CREATE INDEX IF NOT EXISTS idx_audit_entries_timestamp
    ON audit_entries(timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_audit_entries_component
    ON audit_entries(component);

CREATE INDEX IF NOT EXISTS idx_audit_entries_action
    ON audit_entries(action);

-- =============================================================================
-- KNOWLEDGE GAP TRACKING
-- =============================================================================

CREATE TABLE IF NOT EXISTS knowledge_gaps (
    id TEXT PRIMARY KEY,
    trace_id TEXT NOT NULL,
    description TEXT NOT NULL,
    context TEXT,  -- JSON serialized context
    status TEXT NOT NULL CHECK(status IN ('pending', 'processing', 'resolved', 'failed')),
    resolution TEXT,  -- How it was resolved
    code_path TEXT,  -- Path to generated code if resolved
    created_at INTEGER NOT NULL,
    resolved_at INTEGER,
    metadata TEXT  -- JSON serialized metadata
);

CREATE INDEX IF NOT EXISTS idx_knowledge_gaps_status
    ON knowledge_gaps(status);

CREATE INDEX IF NOT EXISTS idx_knowledge_gaps_created_at
    ON knowledge_gaps(created_at DESC);

-- =============================================================================
-- SENSOR FEEDBACK & DISTRIBUTED LEARNING
-- =============================================================================

CREATE TABLE IF NOT EXISTS sensor_feedback (
    id TEXT PRIMARY KEY,
    from_sensor TEXT NOT NULL,
    to_sensor TEXT NOT NULL,
    feedback_type TEXT NOT NULL,
    data TEXT,  -- JSON serialized feedback data
    timestamp INTEGER NOT NULL,
    created_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now') * 1000)
);

CREATE INDEX IF NOT EXISTS idx_sensor_feedback_from
    ON sensor_feedback(from_sensor);

CREATE INDEX IF NOT EXISTS idx_sensor_feedback_to
    ON sensor_feedback(to_sensor);

CREATE INDEX IF NOT EXISTS idx_sensor_feedback_timestamp
    ON sensor_feedback(timestamp DESC);

-- =============================================================================
-- SYSTEM METRICS & MONITORING
-- =============================================================================

CREATE TABLE IF NOT EXISTS system_metrics (
    id TEXT PRIMARY KEY,
    component TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    metric_value REAL NOT NULL,
    timestamp INTEGER NOT NULL,
    metadata TEXT  -- JSON serialized metadata
);

CREATE INDEX IF NOT EXISTS idx_system_metrics_component
    ON system_metrics(component);

CREATE INDEX IF NOT EXISTS idx_system_metrics_name
    ON system_metrics(metric_name);

CREATE INDEX IF NOT EXISTS idx_system_metrics_timestamp
    ON system_metrics(timestamp DESC);

-- =============================================================================
-- VIEWS FOR COMMON QUERIES
-- =============================================================================

-- Recent decisions with high risk
CREATE VIEW IF NOT EXISTS recent_high_risk_decisions AS
SELECT
    id,
    trace_id,
    decision_type,
    reason,
    risk_score,
    datetime(issued_at / 1000, 'unixepoch') as issued_time
FROM kernel_decisions
WHERE risk_score > 0.5
ORDER BY issued_at DESC
LIMIT 100;

-- Cache performance metrics
CREATE VIEW IF NOT EXISTS cache_performance AS
SELECT
    COUNT(*) as total_entries,
    SUM(hit_count) as total_hits,
    AVG(hit_count) as avg_hits_per_entry,
    COUNT(CASE WHEN hit_count > 0 THEN 1 END) as used_entries,
    COUNT(CASE WHEN hit_count = 0 THEN 1 END) as unused_entries
FROM llm_cache;

-- Task success rates
CREATE VIEW IF NOT EXISTS task_performance AS
SELECT
    task_name,
    success_rate,
    execution_count,
    avg_execution_time_ms,
    datetime(created_at / 1000, 'unixepoch') as created_time
FROM learned_tasks
WHERE execution_count > 0
ORDER BY success_rate DESC, execution_count DESC;

-- Belief contradictions
CREATE VIEW IF NOT EXISTS belief_contradictions AS
SELECT
    e.id as edge_id,
    e.source_id,
    n1.content as source_content,
    n1.confidence as source_confidence,
    e.target_id,
    n2.content as target_content,
    n2.confidence as target_confidence,
    e.strength as contradiction_strength
FROM belief_edges e
JOIN belief_nodes n1 ON e.source_id = n1.id
JOIN belief_nodes n2 ON e.target_id = n2.id
WHERE e.type = 'contradicts'
ORDER BY e.strength DESC;

-- =============================================================================
-- INITIALIZATION COMPLETE
-- =============================================================================

-- Insert a marker to indicate schema has been initialized
INSERT OR IGNORE INTO audit_entries (
    id,
    trace_id,
    timestamp,
    component,
    action,
    details
) VALUES (
    'schema_init_' || strftime('%s', 'now'),
    'system',
    strftime('%s', 'now') * 1000,
    'database',
    'schema_initialized',
    '{"version": "0.1.0", "date": "2026-02-03"}'
);
