# Engram 1M Neuron Training Plan

**Date**: 2026-02-21
**Status**: DRAFT — For review before execution
**Depends on**: `deploy/docker-compose.1m.yml` (deployment config)

---

## 1. The Wall-Time Problem

The existing Hetzner plan has an **inconsistency** that must be addressed before deploying.

### Default Phase Boundaries (CriticalPeriodConfig)

| Phase | Ends At | Steps In Phase |
|-------|---------|---------------|
| Infant | 600,000 | 600,000 |
| Toddler | 3,600,000 | 3,000,000 |
| Juvenile | 21,600,000 | 18,000,000 |
| Adolescent | Dynamic entry | Experience-dependent |

### Wall Time at 1M Neurons (0.3-1.0 steps/sec on CCX63)

| Phase | Steps | @ 1.0 steps/sec | @ 0.3 steps/sec |
|-------|-------|-----------------|-----------------|
| Infant | 600K | **7 days** | **23 days** |
| Toddler | 3.6M | 42 days | 139 days |
| Juvenile | 21.6M | 250 days | 833 days |

**Problem**: At best-case 1.0 steps/sec, reaching toddler-end takes 42 days ($171). The Hetzner plan budgets 3-7 days ($12-29). Even reaching infant-end (600K steps) takes 7 days at best case.

The existing plan's training schedule (line 338) says "T+36hr → may transition to toddler at ~600K steps" — this assumes ~4.6 steps/sec, which is the rate for **220K neurons**, not 1M.

### Solution: Make Phase Boundaries Env-Configurable

Add `NEURO_INFANT_END`, `NEURO_TODDLER_END`, `NEURO_JUVENILE_END` env vars to `config.py:NeuromorphicConfig.from_env()`. Use compressed boundaries for the PoC:

| Phase | Default | PoC (10x compressed) | Wall Time @ 0.5 steps/sec |
|-------|---------|---------------------|--------------------------|
| Infant | 600,000 | **60,000** | ~33 hours |
| Toddler | 3,600,000 | **360,000** | ~8.3 days |
| Juvenile | 21,600,000 | **2,160,000** | ~50 days |

**With 10x compression, a 5-day run ($20) reaches mid-toddler phase**, demonstrating:
- Full infant phase with wide-open plasticity (DA=2.0)
- Toddler transition with guided exploration (DA=1.5)
- Measurable phase-dependent behavioral change

This does NOT change the science — the same learning rules, neuromodulator levels, and STDP dynamics all function identically. Only the step count at which phases transition is shortened. The brain still gets the same amount of *input* — it just transitions phases faster.

---

## 2. What Needs to Be Implemented

### Code Changes (Required)

| # | Change | File | Effort | Why |
|---|--------|------|--------|-----|
| 1 | **Env-configurable phase boundaries** | `config.py` | 30 min | Without this, 5-day run stays in infant phase |
| 2 | **Create `docker-compose.1m.yml`** | `deploy/` | 15 min | Overlay config for 1M neuron deployment |
| 3 | **Phase-appropriate video curriculum** | `deploy/download-videos-1m.sh` | 1 hour | Current 7 videos are all baby-level, no progression |
| 4 | **Export CNN ONNX model** | `sensory-gateway/models/` | 15 min | Gateway needs MobileNetV3 for CNN retina mode |

### Code Changes (Optional, recommended)

| # | Change | File | Effort | Why |
|---|--------|------|--------|-----|
| 5 | Add step-rate logging to service loop | `service.py` | 15 min | Needed to verify actual steps/sec on Hetzner |
| 6 | Add phase transition NATS events | `neuromodulation.py` | 15 min | Dashboard can show developmental milestones |

**Total required effort: ~2 hours**

---

## 3. Video Curriculum Design

The brain's developmental phases mirror human infant development. Videos should match what each phase needs:

### Phase → Content Mapping

| Phase | Neuromodulator Profile | Learning Goal | Video Content |
|-------|----------------------|---------------|---------------|
| **Infant** (DA=2.0, ACh=2.5) | Maximum plasticity, absorb everything | Basic visual/auditory encoding, raw pattern detection | High-contrast, simple objects, single items on screen, slow pace, repetitive |
| **Toddler** (DA=1.5, ACh=1.5) | Guided exploration, category formation | Category boundaries, multi-object scenes | Category flashcards (animals, colors, shapes), clear labels, moderate pace |
| **Juvenile** (DA=1.2, ACh=1.2) | Refinement, prediction | Fine discrimination, sequence learning | Complex scenes, action sequences, cause-and-effect, counting |

### Video Curriculum (28 videos, ~3 hours total)

**Tier 1: Infant Phase (10 videos, played first)**
Focus: High-contrast, single objects, slow, repetitive

| # | Category | Content Type | Example Search |
|---|----------|-------------|----------------|
| 1 | Sensory | Black & white high-contrast patterns for babies | "high contrast baby visual stimulation" |
| 2 | Sensory | Colored shapes moving slowly | "baby sensory videos shapes" |
| 3 | Sensory | Single objects with labels (one at a time) | "baby first words one word at a time" |
| 4 | Animals | Single animal, name, sound, repeat | "animal sounds for babies one at a time" |
| 5 | Objects | Household objects one by one | "first words baby objects" |
| 6 | Colors | Single color fills screen, color name | "learn colors for babies simple" |
| 7 | Food | Single fruit/vegetable, name, repeat | "fruits for babies flashcards" |
| 8-10 | Mixed | First 50 words style (existing videos) | (keep 3 of the existing 7) |

**Tier 2: Toddler Phase (10 videos, queued after infant convergence)**
Focus: Categories, multiple items, moderate pace, comparison

| # | Category | Content Type | Example Search |
|---|----------|-------------|----------------|
| 11 | Animals | Farm animals with sounds (multiple) | "farm animals for toddlers" |
| 12 | Animals | Wild/safari animals | "wild animals for kids" |
| 13 | Shapes | Shapes with real-world examples | "shapes in real life for toddlers" |
| 14 | Colors | Colors in context (red apple, blue sky) | "colors in real life toddlers" |
| 15 | Body | Body parts with actions | "body parts for toddlers" |
| 16 | Vehicles | Vehicles with sounds | "vehicles for toddlers with sounds" |
| 17 | Numbers | Counting 1-10 with objects | "counting for toddlers" |
| 18 | Actions | Action words (run, jump, eat) | "action words for toddlers" |
| 19 | Food | Meals (breakfast items, lunch items) | "food vocabulary for kids" |
| 20 | Objects | Household items in rooms | "household objects for kids" |

**Tier 3: Juvenile Phase (8 videos, queued after toddler convergence)**
Focus: Complex scenes, sequences, relationships, prediction

| # | Category | Content Type | Example Search |
|---|----------|-------------|----------------|
| 21 | Sequences | Daily routine (wake up, eat, play, sleep) | "daily routine for kids" |
| 22 | Cause/Effect | Simple cause and effect scenes | "cause and effect for kids" |
| 23 | Categories | "Which one doesn't belong?" | "odd one out for kids" |
| 24 | Numbers | Counting 1-20, simple addition | "counting to 20 for kids" |
| 25 | Actions | Multi-step actions (cooking, building) | "step by step activities for kids" |
| 26 | Animals | Animal habitats (where animals live) | "animal habitats for kids" |
| 27 | Spatial | Prepositions (in, on, under, next to) | "prepositions for kids" |
| 28 | Social | Emotions/expressions | "emotions for kids" |

### Curriculum Delivery Strategy

The gateway's `--video-loop` flag loops all provided videos. For phase-appropriate feeding:

**Manual phase transitions** (simplest, recommended for PoC):
1. Start with Tier 1 videos only: `--video /data/videos/tier1/*.mp4 --video-loop`
2. Monitor dashboard. When infant→toddler transition occurs, stop gateway
3. Re-launch with Tier 1 + Tier 2: `--video /data/videos/tier1/*.mp4 /data/videos/tier2/*.mp4 --video-loop`
4. Repeat for Tier 3 at juvenile transition

**Why manual**: Automated curriculum switching would require new code (gateway watching NATS for phase events). Manual switching is sufficient for a 5-day PoC and avoids bugs.

---

## 4. Revised Training Schedule

### With 10x Compressed Phase Boundaries

Assuming 0.5 steps/sec average (midpoint of expected 0.3-1.0 range):

| Time | Steps | Phase | Action |
|------|-------|-------|--------|
| T+0 | 0 | Boot | Start services, verify 1,001,800 neurons |
| T+15min | 0 | Ready | Start gateway with Tier 1 videos |
| T+1hr | ~1,800 | Infant | First backup, verify STDP deltas > 0 |
| T+12hr | ~21,600 | Infant | Backup, check convergence trend |
| T+33hr | **60,000** | **→ Toddler** | Phase transition! Switch to Tier 1+2 videos |
| T+48hr | ~86,400 | Toddler | Backup, verify DA dropped to 1.5 |
| T+72hr | ~129,600 | Toddler | Check category differentiation |
| T+96hr | ~172,800 | Toddler | Backup, assess convergence |
| T+120hr | ~216,000 | Toddler | **5-day stop point — $20** |

**What this achieves:**
- Full infant phase (33 hours) with maximum plasticity
- ~60% of toddler phase with guided learning
- At least one developmental phase transition (investor demo criterion)
- Category-differentiated inputs across phases

### Extended Run (7 days, $29)

| Time | Steps | Phase | Action |
|------|-------|-------|--------|
| T+120hr | ~216,000 | Toddler | Continue |
| T+138hr | **360,000** | **→ Juvenile** | Phase transition! Add Tier 3 videos |
| T+168hr | ~302,400 | Juvenile | **7-day stop — $29** |

A 7-day run gets to early juvenile phase — two phase transitions demonstrated.

---

## 5. Success Criteria (Revised)

### Minimum Success — "It scales" ($12, 3 days)
- [ ] 1,001,800 neurons running stably for 72 hours
- [ ] No OOM kills
- [ ] STDP weight updates active on all 24 synapse groups
- [ ] STDP delta decreasing (learning signal)
- [ ] Successful save → shutdown → reload cycle

### Good Success — "It learns and develops" ($20, 5 days)
All minimum, plus:
- [ ] Infant → toddler phase transition occurs
- [ ] Neuromodulator baselines shift (DA: 2.0 → 1.5)
- [ ] Different video categories produce distinguishable weight patterns
- [ ] Convergence rate improves on repeated video exposure
- [ ] Prediction error decreases over time

### Excellent Success — "Investor demo" ($29, 7 days)
All good, plus:
- [ ] Two phase transitions (infant → toddler → juvenile)
- [ ] Concept layer shows stable k-WTA winner patterns
- [ ] Trained brain responds differently to known vs novel inputs (tested locally after download)
- [ ] Feature layer shows visual feature selectivity
- [ ] Cognitive action channel fires for genuinely unfamiliar inputs

---

## 6. Execution Checklist

### Before Server (local, today)

- [ ] **Code change**: Add env-configurable phase boundaries to `config.py`
- [ ] **Code change**: Create `deploy/docker-compose.1m.yml` overlay config
- [ ] **Script**: Create `deploy/download-videos-1m.sh` with tiered video curriculum
- [ ] **Script**: Export MobileNetV3 ONNX model for CNN retina
- [ ] **Tests**: Run full test suite — all tests must pass
- [ ] **Verify**: `docker compose -f docker-compose.yml -f deploy/docker-compose.1m.yml config` parses correctly

### On Server (deployment day)

1. Provision Hetzner CCX63 (192 GB RAM, 48 vCPU)
2. Run `cloud-init.sh` + add 16 GB swap
3. `sync.sh` codebase to server
4. Download Tier 1 videos (10 videos)
5. Build Docker images
6. Start nats → ollama → dashboard → neuromorphic → cognitive-bridge
7. Verify "Network ready: 1,001,800 neurons"
8. Start gateway with Tier 1 videos + `--turbo --cnn --video-fps 2`
9. Verify STDP deltas > 0 within 1 hour
10. Set up auto-backup every 12 hours
11. Set auto-shutdown cron at 168 hours

### During Training

- Monitor every 12 hours via dashboard + deploy/monitor.sh
- Backup every 12 hours
- Switch video tiers at phase transitions
- Log step rate, phase, STDP deltas at each check

### After Training

1. Force WAL checkpoint
2. Download trained brain SQLite
3. Delete server
4. Load locally for analysis
5. Document results

---

## 7. What This Plan Does NOT Include

These are deferred until after the PoC training:

- **Pattern Separator region** — additive, trains on top of existing weights
- **Oscillatory dynamics** — risky to firing patterns, add after baseline established
- **Izhikevich/AdEx neurons** — changes dynamics but preserves synaptic weights
- **GPU CuPy acceleration** — identical math, just faster; would help but not required
- **Automated curriculum switching** — manual is fine for PoC

All of these stack on top of trained weights (confirmed in previous analysis). The PoC establishes a trained baseline that future features build upon.

---

## 8. Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Steps/sec < 0.3 | Barely reach infant end in 5 days | Reduce to 500K neurons, or increase STDP interval to 5 |
| 10x compressed phases too fast | Brain doesn't stabilize within phases | Try 5x compression instead (infant=120K steps) |
| Video downloads blocked | No training content | Download locally first, scp to server |
| Convergence never triggers | Curriculum auto-advance doesn't fire | Manual video tier switching is independent of convergence |
| OOM during peak STDP | Service crash | 16 GB swap + 50G Docker limit + 30s save interval |

---

*This plan requires your review before execution. The key decision: do you approve the 10x compressed phase boundaries for the PoC?*
