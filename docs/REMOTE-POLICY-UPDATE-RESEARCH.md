# Remote AI Policy Update Research for Safety-Critical Edge Systems

**Research Date:** March 15, 2026
**Context:** Engram spiking neural network brain running on edge robots
**Goal:** Remote adjustment of safety beliefs, body profiles, and behavioral constraints without dangerous transients

---

## Executive Summary

This research examines how autonomous vehicles, robotics systems, and military applications handle remote AI behavior updates in safety-critical contexts. Key findings:

1. **Shadow mode validation** is the gold standard for testing AI updates before deployment
2. **Gradual rollout strategies** (canary deployments) minimize risk across fleets
3. **Regulatory frameworks** (ISO 21434, UNECE WP.29) mandate formal software update management systems
4. **Digital twins** enable pre-deployment testing in simulated environments
5. **Multi-layered safety** combines geofencing, ODD monitoring, force limits, and kill switches
6. **Policy switching transients** remain an active research area with no universal solution

---

## 1. Autonomous Vehicle OTA Updates

### Tesla Autopilot Safety Framework

**Deployment Strategy:**
- Phased rollout: employees → early access → high-safety-score drivers → general release
- Shadow mode testing: new models process live data but don't control vehicle, compare decisions with human driver
- FSD v14.2.2.5 (Feb 2026): Higher-resolution vision encoders across 8 cameras for subtle cue detection

**Key Insight:** Tesla transparently maintains "Supervised" designation where driver remains responsible, critical for both NHTSA (US) and European regulators (DVLA, RDW).

**Sources:**
- [Tesla Autopilot in 2026: Options, Features, and What To Expect](https://www.diywrapclub.com/a/blog/tesla-autopilot-in-2026-options-features-and-what-drivers-can-expect)
- [Hands-Off Evolution – FSD v14 and the 2026.2.9.1 OTA Update](https://www.teslaacessories.com/blogs/news/hands-off-evolution-%E2%80%93-a-comprehensive-deep-dive-into-fsd-v14-and-the-2026.2.9.1-ota-update)
- [Tesla's FSD Shadow Mode: What It Is and How It Improves FSD](https://www.notateslaapp.com/news/3108/teslas-fsd-shadow-mode-what-it-is-and-how-it-improves-fsd)

### Waymo Software Update & Rollback Mechanisms

**Update Delivery:**
- OTA updates deployed fleet-wide in 3-week windows (e.g., Dec 20 → Jan 12)
- Wireless updates similar to smartphone patches, no service center visits required
- Service continues uninterrupted during update deployment

**Rollback Examples:**
- **School bus detection failure:** Software update deployed after incidents
- **Towed vehicle recognition:** Prediction error for towed trucks (2022-2024) fixed via targeted update
- **Object detection:** Failed to detect thin/semi-stationary objects, 7 low-speed crashes → software repair

**Regulatory Compliance:**
- US Code Title 49, section 30118: Must notify NHTSA even if fix already deployed
- Voluntary recalls filed for software defects
- Service not interrupted by safety updates

**Sources:**
- [Waymo updating software after self-driving cars passed stopped school buses](https://www.npr.org/2025/12/06/nx-s1-5635614/waymo-school-buses-recall)
- [Waymo updated its vehicle software after 'rare scenario' in Phoenix](https://www.smartcitiesdive.com/news/waymo-recalls-software-nhtsa-tow-truck-phoenix-autonomous-vehicles/707923/)
- [Voluntary recall of our previous software](https://waymo.com/blog/2024/02/voluntary-recall-of-our-previous-software)
- [Waymo speeds up safety, software updates ahead of 2026 expansion](https://news.dealershipguy.com/p/waymo-speeds-up-safety-software-updates-ahead-of-2026-expansion-2025-12-29)

---

## 2. Safe Policy Switching for Neural Networks

### Research Findings

**Core Problem:**
Intermittent and frequent switching between trained policy and safety controller results in undesirable behaviors and reduced performance.

**Solutions Under Development:**

1. **Runtime Monitoring & Repair:**
   - Monitor neural network + certificate functions to detect property violations
   - Extract new training data from violations
   - Re-train and repair policy + certificate function
   - Source: [Neural Control and Certificate Repair via Runtime Monitoring](https://arxiv.org/html/2412.12996)

2. **Policy Repair (Minimize Switching):**
   - Repair trained policy using runtime data from safety controller
   - Deviate minimally from original policy
   - Reduce/eliminate control switching frequency
   - Source: [Runtime-Safety-Guided Policy Repair](https://arxiv.org/abs/2008.07667)

3. **Safe Online Controller Switching:**
   - Switch between NNCS controller and obstacle avoidance based on verification results
   - Verification-driven decision making
   - Source: [Case Study: Runtime Safety Verification of Neural Network Controlled System](https://arxiv.org/abs/2408.08592)

4. **Formal Methods Integration:**
   - Integrate formal verification into learning process
   - Provide safety guarantees across design, training, execution lifecycle
   - Source: [Safeguarding Neural Network-Controlled Systems via Formal Methods](https://link.springer.com/chapter/10.1007/978-3-031-98208-8_1)

**Key Insight:** Policy switching transients remain an unsolved challenge. Current approaches focus on minimizing switching frequency through repair and formal verification rather than eliminating transients entirely.

**Sources:**
- [Neural Control and Certificate Repair via Runtime Monitoring](https://ojs.aaai.org/index.php/AAAI/article/view/34840/36995)
- [Runtime-Safety-Guided Policy Repair](https://link.springer.com/chapter/10.1007/978-3-030-60508-7_7)
- [Case Study: Runtime Safety Verification of Neural Network Controlled System](https://arxiv.org/abs/2408.08592)

---

## 3. Edge AI Rollback Mechanisms

### Core Capabilities

**Definition:**
AI rollback allows systems to revert to previous stable states when updates fail. Critical for edge deployments where connectivity cannot be guaranteed and local systems must make autonomous safety decisions.

**Key Features:**

1. **Delta Updates & Rollbacks:**
   - Delta packaging reduces bandwidth load
   - Robust rollback for failed deployments
   - OTA model updates without physical access
   - Source: [Edge AI Model Lifecycle Management](https://aithority.com/machine-learning/edge-ai-model-lifecycle-management-versioning-monitoring-and-retraining/)

2. **Distributed Version Control:**
   - Track model performance across edge locations
   - Enable rapid rollback when issues arise
   - Autonomous rollback for edge devices with limited connectivity
   - Source: [Versioning, Rollback & Lifecycle Management of AI Agents](https://medium.com/@nraman.n6/versioning-rollback-lifecycle-management-of-ai-agents-treating-intelligence-as-deployable-deac757e4dea)

3. **Federated Rollback:**
   - Coordinated but autonomous rollback across multiple devices
   - Respond to local conditions while maintaining global consistency
   - Source: [Hitting the Undo Button: The Critical Role of Rollback in AI Systems](https://www.sandgarden.com/learn/rollback)

### Performance Considerations

**Overhead Challenges:**
- Continuous monitoring performance impact
- Storage requirements for multiple versions
- Network bandwidth for rapid rollback
- Computational resources for real-time anomaly detection
- Particularly critical for edge/resource-constrained environments

**Detection & Response:**
- Effective monitoring recognizes performance degradation
- Triggers retraining or rollback before costly production errors
- Health checks at each deployment stage
- One-click rollback if issues arise

**Sources:**
- [Hitting the Undo Button: The Critical Role of Rollback in AI Systems](https://www.sandgarden.com/learn/rollback)
- [Edge AI Model Lifecycle Management](https://aithority.com/machine-learning/edge-ai-model-lifecycle-management-versioning-monitoring-and-retraining/)
- [Rollback Mechanisms for Autonomous Code Changes: A Comprehensive Review](https://mgx.dev/insights/rollback-mechanisms-for-autonomous-code-changes-a-comprehensive-review/1c707a9f8345475dba35b5b91f979191)

---

## 4. Military Drone Remote Rule-of-Engagement Updates

### Current Framework

**US Military Standing Rules of Engagement (CJCSI 3121.01B):**
- Unit commanders retain inherent right and obligation to exercise unit self-defense
- Response authorized for hostile act or demonstrated hostile intent
- Example: Aircraft may use force if illuminated by fire-control anti-aircraft radar
- May deploy anti-radiation missile to destroy radar and negate immediate threat

**Positive Identification (PID) Requirements:**
- Soldier must positively identify individual as threat before engagement
- For drone strikes, PID becomes even more critical:
  - Required to authorize mission
  - Required to verify mission success
- Source: [Drones and the Standing Rules of Engagement Regarding Self-Defense](https://www.lawfaremedia.org/article/drones-and-standing-rules-engagement-regarding-self-defense)

### Adaptation Challenges

**Key Issue:** Rules of Engagement require interpretation/rewriting for widespread drone use. Current frameworks designed for traditional manned aircraft.

**Gap:** Search results indicate no publicly documented remote ROE update mechanisms for deployed military drones. This is likely classified or not publicly discussed for operational security reasons.

**Sources:**
- [Drones and the Standing Rules of Engagement Regarding Self-Defense](https://www.lawfaremedia.org/article/drones-and-standing-rules-engagement-regarding-self-defense)
- [Drone Defense at Home: Closing the CUAS Rules of Engagement Gap](https://www.commercialuavnews.com/drone-defense-at-home-closing-the-cuas-rules-of-engagement-roe-gap)

---

## 5. Digital Twin Architectures for Policy Testing

### Robotics Applications

**Core Concept:**
Digital twins reflect real system in digital environment, allowing robot policies to be developed precisely and readily transmitted to physical robot.

**Key Architectures:**

1. **Modular Digital Twin Framework for Collaborative Robotics:**
   - Scalable representations of cyber-physical environments
   - Tools for safety analysis and control
   - Investigate safety within collaborative manufacturing
   - Source: [A Modular Digital Twinning Framework for Safety Assurance of Collaborative Robotics](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2021.758099/full)

2. **AI-Enabled Digital Twin Systems:**
   - Real-time monitoring
   - Predictive maintenance
   - Intelligent process optimization
   - Generative AI + Predictive AI modules
   - Source: [Generative and Predictive AI for digital twin systems in manufacturing](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1655470/full)

3. **Sim2Real Training & Testing:**
   - Generate virtual training data for deep learning
   - Test agent policies using reinforcement learning
   - Include human element in virtual environment
   - Source: [Towards next generation digital twin in robotics](https://pmc.ncbi.nlm.nih.gov/articles/PMC9941953/)

### Dynamic Reconfiguration Example

**Autonomous Robot Controller Reconfiguration:**
- Virtual replica of robot's operational environment
- Simulate and optimize movement trajectories in response to real-world changes
- Recalculate paths and control parameters
- Deploy updated code to physical robot
- Source: [Digital Twin based Automatic Reconfiguration of Robotic Systems](https://arxiv.org/html/2511.00094v1)

**Key Insight:** Digital twins enable risk-free policy testing before physical deployment, critical for safety-critical systems.

**Sources:**
- [Towards next generation digital twin in robotics: Trends, scopes, challenges, and future](https://pmc.ncbi.nlm.nih.gov/articles/PMC9941953/)
- [A Modular Digital Twinning Framework for Safety Assurance of Collaborative Robotics](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2021.758099/full)
- [Digital twins to embodied artificial intelligence: review and perspective](https://www.oaepublish.com/articles/ir.2025.11)
- [AI and digital twins to serve increasingly complex robot management](https://www.computerweekly.com/feature/AI-and-digital-twins-to-serve-increasingly-complex-robot-management)

---

## 6. Federated Learning: Edge-to-Cloud Aggregation

### Architecture Overview

**Core Concept:**
Federated learning (FL) enables collaborative model training across distributed devices while preserving data privacy. Raw data remains on edge devices; only model updates shared with central server.

**Cloud-Edge Collaboration:**
- Workload dispersed among cloud, base stations, edge nodes, end devices
- Each performs small portion of work
- Results aggregated centrally
- FL server uses aggregation algorithms (e.g., FedAvg) to generate new global model
- Source: [Federated learning in cloud-edge collaborative architecture](https://link.springer.com/article/10.1186/s13677-022-00377-4)

### Implementation Patterns

1. **Two-Level Aggregation (HybridFL):**
   - Edge layer aggregation
   - Cloud layer aggregation
   - Asynchronous global updates
   - Source: [Federated learning in cloud-edge collaborative architecture](https://pmc.ncbi.nlm.nih.gov/articles/PMC9753079/)

2. **Robotic Fleet Applications:**
   - UAVs act as both learners and contributors
   - Develop efficient decentralized model aggregation
   - Federated deep RL for task scheduling in heterogeneous robotic fleets
   - Optimize logistics operations
   - Source: [Federated learning at the edge in Industrial Internet of Things](https://www.sciencedirect.com/science/article/pii/S2210537925000071)

### Challenges for Edge Deployment

**Scalability & Communication Overhead:**
- Major bottlenecks for Industrial IoT applications
- Limited communication bandwidth
- Computational constraints of edge devices
- Maintaining low latency + high accuracy becomes challenging
- Source: [Federated learning at the edge in Industrial Internet of Things](https://www.sciencedirect.com/science/article/pii/S2210537925000071)

**Key Insight:** Federated learning enables fleet-wide learning aggregation but requires careful bandwidth and compute management for edge robotics.

**Sources:**
- [Federated learning in cloud-edge collaborative architecture: key technologies, applications and challenges](https://link.springer.com/article/10.1186/s13677-022-00377-4)
- [Federated learning at the edge in Industrial Internet of Things: A review](https://www.sciencedirect.com/science/article/pii/S2210537925000071)
- [A Review on Federated Learning Architectures for Privacy-Preserving AI](https://www.mdpi.com/2079-9292/14/13/2512)
- [Federated Learning in Edge Computing: A Systematic Survey](https://pmc.ncbi.nlm.nih.gov/articles/PMC8780479/)

---

## 7. Regulatory Frameworks for OTA Updates

### ISO 21434: Automotive Cybersecurity

**Overview:**
- Published August 2021 (ISO/SAE 21434:2021)
- International standard for automotive cybersecurity engineering
- Targets electrical/electronic (E/E) systems in vehicles
- Covers entire vehicle lifecycle: concept → development → production → operation → maintenance → decommissioning

**OTA Update Requirements:**
- Campaign management
- OTA software update delivery
- In-vehicle flashing
- **Critical:** Software updates can be attack vectors, require own cyber risk management
- Source: [Cybersecurity in Automotive OTA Update Systems and Automotive Software Stores](https://saemobilus.sae.org/papers/cybersecurity-automotive-ota-update-systems-automotive-software-stores-2026-26-0621)

**Sources:**
- [Automotive cybersecurity: ISO/SAE 21434](https://www.appliedintuition.com/blog/iso-sae-21434-shaping-automotive-cybersecurity)
- [What is ISO 21434? — The New Standard for Automotive Cybersecurity](https://autosar.io/en/insights/iso21434-cybersecurity-guide)
- [How to comply with ISO/SAE 21434 & UNECE R155/R156](https://mender.io/blog/how-to-comply-with-iso-sae-21434-and-unece-r155-r156-for-robust-automotive-cybersecurity)

### UNECE WP.29 Regulations

**R155: Cyber Security Management System (CSMS)**
- Mandatory from July 2024 for all vehicles in 54 countries (EU, UK, Japan, South Korea)
- Comprehensive CSMS throughout vehicle lifecycle
- Applies to vehicles with autonomous driving functions from Level 3

**R156: Software Update Management System (SUMS)**
- Ensures software updates carried out safely and in legal compliance
- Guarantees safety of software in vehicle control systems throughout lifecycle
- Requires framework for:
  - Recording hardware/software versions
  - Assessing if updates affect type approval parameters
  - Determining if updates affect safety or safe driving

**Key Insight:** Both regulations apply to Level 3+ autonomous vehicles. Manufacturers must implement formal CSMS and SUMS.

**Sources:**
- [UNECE WP.29 R155/R156: new cybersecurity regulations for vehicles](https://www.appluslaboratories.com/global/en/news/publications/new-cybersecurity-regulations-vehicles-unece-wp29)
- [Compliance with UN R156: Securing Vehicle Software Updates](https://autocrypt.io/unr156-securing-vehicle-software-updates/)
- [Automotive Cybersecurity Regulation - UNECE R155](https://upstream.auto/blog/automotive-cybersecurity-regulation-unece-wp29-r155/)
- [UN Regulations on Cybersecurity and Software Updates to pave the way for mass roll out of connected vehicles](https://unece.org/sustainable-development/press/un-regulations-cybersecurity-and-software-updates-pave-way-mass-roll)

---

## 8. Remote Safety Parameter Adjustment

### Force Limits

**Power and Force Limitation (PFL):**
- Reduces effects of unintended human-robot contact
- Control schemes ensure forces and momentum upon impact within safe limits
- Confines torque command within preset safety limit
- Source: [A survey of safety control for service robots](https://www.sciencedirect.com/science/article/pii/S2949855425000589)

**Advanced Approaches:**
- Novel PFL formally guarantees human safety for:
  - Dynamically changing robot paths
  - Full human-robot contact
  - Arbitrary human motion
  - Sharp robot geometries
- Source: [A General Safety Framework for Autonomous Manipulation in Human Environments](https://arxiv.org/html/2412.10180)

### Dynamic Safety Adjustments

**Pre-Collision Control:**
- Monitor human, robot, or both
- Modify robot control parameters prior to collision/contact
- Source: [Robot Safety - an overview](https://www.sciencedirect.com/topics/engineering/robot-safety)

**Context-Aware Thresholds:**
- Rather than single global speed/force limit, adjust based on:
  - Task type
  - Tooling
  - Worker proximity
- Example: Robot sanding panel runs full-speed when alone, automatically slows when technician enters work area
- Source: [Collaborative robot safety standards you must know](https://standardbots.com/blog/collaborative-robot-safety-standards)

**Future Standards:**
- Robots predict unsafe movement
- Adjust thresholds based on tasks
- Integrate with plant-wide safety systems

**Sources:**
- [A survey of safety control for service robots](https://www.sciencedirect.com/science/article/pii/S2949855425000589)
- [Collaborative robot safety standards you must know](https://standardbots.com/blog/collaborative-robot-safety-standards)
- [A General Safety Framework for Autonomous Manipulation in Human Environments](https://arxiv.org/html/2412.10180)

---

## 9. Geofencing and Operational Design Domain (ODD) Management

### Operational Design Domain (ODD)

**Definition:**
Set of environmental conditions an autonomous system is designed to work in. Context defined by:
- Environmental conditions
- Geographical boundaries
- Time of day
- Other operational conditions

**Source:** [Operational design domain - Wikipedia](https://en.wikipedia.org/wiki/Operational_design_domain)

### Geofencing's Role

**Basic Approach:**
- Virtual perimeters describing real geographic areas
- Simplest way to define ODD and monitor operational domain in real-time
- Weather conditions approximation (rain, snow, sun)
- Source: [Safe Autonomy: Operational Design Domain for Autonomous Systems](http://safeautonomy.blogspot.com/2019/06/operational-design-domain-odd-for.html)

**Limitations:**
Geofencing alone is insufficient. Modern systems require:
- Attribute-rich operational domain monitoring beyond conventional geofencing
- Exit monitors checking local operating conditions
- Improves system performance for complex ODDs
- Source: [Overview of the Operational Design Domain Monitoring](https://hal.science/hal-04402955v1/document)

### ODD Monitoring for Safety

**Real-Time Assessment:**
- Determine if current operational domain compatible with designed/validated ODD
- Crucial for maintaining safe autonomous operations
- More sophisticated than simple geofencing

**Key Insight:** While geofencing provides foundational ODD management, modern implementations require multi-attribute monitoring beyond geographic boundaries.

**Sources:**
- [Safe Autonomy: Operational Design Domain for Autonomous Systems](http://safeautonomy.blogspot.com/2019/06/operational-design-domain-odd-for.html)
- [Overview of the Operational Design Domain Monitoring](https://hal.science/hal-04402955v1/document)
- [Operational design domains | Autonomous Vehicle Systems Class Notes](https://fiveable.me/autonomous-vehicle-systems/unit-1/operational-design-domains/study-guide/nawsz63xLm0ecxJB)
- [Defining ODD boundaries in 2 phases](https://www.appliedintuition.com/blog/odd-taxonomy-a-2-phase-approach-to-defining-testing-boundaries)

---

## 10. Human Oversight Systems

### Kill Switch Mechanisms

**Definition:**
Mechanism to shut down or override autonomous systems in conflict/dangerous situations.

**Implementation Forms:**

1. **Manual Override:**
   - Simple accessible operator control
   - Immediate shutdown capability

2. **Automated Monitoring:**
   - System monitors agent behavior for predefined risky patterns
   - Triggers shutdown automatically
   - Source: [The 'Kill Switch' for AI Agents: Keeping Autonomous Systems in Check](https://www.oreateai.com/blog/the-kill-switch-for-ai-agents-keeping-autonomous-systems-in-check/37dde1a7a6cb2d24c594ef20877bc8ec)

3. **Layered Shutdown Systems:**
   - Real-time scope enforcement
   - Not static permission boundaries
   - Multiple layers of control
   - Source: [Trustworthy AI Agents: Kill Switches and Circuit Breakers](https://www.sakurasky.com/blog/missing-primitives-for-trustworthy-ai-part-6/)

### Dashboard and Alert Systems

**Interface Requirements:**
- Present information clearly and accessibly
- Enable operators to be promptly alerted
- Comprehend system status
- Make informed decisions
- Source: [TechDispatch #2/2025 - Human Oversight of Automated Decision-Making](https://www.edps.europa.eu/data-protection/our-work/publications/techdispatch/2025-09-23-techdispatch-22025-human-oversight-automated-making_en)

**Dashboard Features:**
- Key findings sent to appropriate stakeholders
- All data on intuitive AI risk dashboard
- Continuous visibility
- Unified governance
- Streamlined oversight across organization
- Source: [Why Every Agent Needs a Kill Switch (And What We Built)](https://www.clawctl.com/blog/ai-agent-kill-switch)

### Alert Management

**Root-Cause Analysis:**
When automated decision-making errors occur:
- Examine design of human oversight tools
- Review appropriateness of alert thresholds
- Assess performance of communication channels under stress
- Source: [TechDispatch #2/2025 - Human Oversight of Automated Decision-Making](https://www.edps.europa.eu/data-protection/our-work/publications/techdispatch/2025-09-23-techdispatch-22025-human-oversight-automated-making_en)

**Visibility Requirements:**
- Kill switch state visible on operator dashboard
- Breaker states displayed
- Real-time system status
- Source: [Trustworthy AI Agents: Kill Switches and Circuit Breakers](https://www.sakurasky.com/blog/missing-primitives-for-trustworthy-ai-part-6/)

**Key Insight:** Modern autonomous systems require multi-layered control combining manual overrides, automated monitoring, clear dashboard visualization, and thoughtful alert threshold design.

**Sources:**
- [Why Every Agent Needs a Kill Switch (And What We Built)](https://www.clawctl.com/blog/ai-agent-kill-switch)
- [The 'Kill Switch' for AI Agents: Keeping Autonomous Systems in Check](https://www.oreateai.com/blog/the-kill-switch-for-ai-agents-keeping-autonomous-systems-in-check/37dde1a7a6cb2d24c594ef20877bc8ec)
- [TechDispatch #2/2025 - Human Oversight of Automated Decision-Making](https://www.edps.europa.eu/data-protection/our-work/publications/techdispatch/2025-09-23-techdispatch-22025-human-oversight-automated-making_en)
- [Trustworthy AI Agents: Kill Switches and Circuit Breakers](https://www.sakurasky.com/blog/missing-primitives-for-trustworthy-ai-part-6/)

---

## 11. Deployment Strategies: Shadow Mode, Blue-Green, Canary

### Shadow Mode Validation

**How It Works:**
- New AI models process live sensory data alongside operational system
- Does NOT control vehicle/robot
- System on standby while vehicle/robot operates normally
- Receives sensor data, makes decisions in real-time
- Compares decisions with actual operator decisions
- Different decisions sent back for analysis
- Source: [Shadow Testing in Autonomous Vehicles: A Novel Approach](https://www.researchgate.net/publication/385733470_Shadow_Testing_in_Autonomous_Vehicles_A_Novel_Approach_to_Validating_Full_Self-Driving_AI_Systems)

**Benefits:**
- Validates new FSD AI models in real-world conditions without compromising safety
- Collects valuable performance metrics
- Identifies edge cases
- Mitigates risks of direct deployment
- Continuous feedback loop for iterative improvement
- Source: [Tesla's FSD Shadow Mode: What It Is and How It Improves FSD](https://www.notateslaapp.com/news/3108/teslas-fsd-shadow-mode-what-it-is-and-how-it-improves-fsd)

**Challenges:**
- Duplicate inference workloads (2-3x compute resources)
- Significant storage for detailed logging
- Open-loop limitation: Cannot observe effects of different actions
- Source: [Shadow Mode Testing | Wiki](https://beta.hyper.ai/en/wiki/32774)

**Industry Adoption:**
First proposed by Tesla, now regarded as key weapon for companies taking "progressive" approach to leverage data advantages.

**Sources:**
- [Shadow Testing in Autonomous Vehicles : A Novel Approach to Validating Full Self-Driving AI Systems](https://www.researchgate.net/publication/385733470_Shadow_Testing_in_Autonomous_Vehicles_A_Novel_Approach_to_Validating_Full_Self-Driving_AI_Systems)
- [AI Shadow Mode: How Systems Learn From You Without Interfering](https://quickgenai.in/ai-shadow-mode)
- [Tesla's FSD Shadow Mode: What It Is and How It Improves FSD](https://www.notateslaapp.com/news/3108/teslas-fsd-shadow-mode-what-it-is-and-how-it-improves-fsd)

### Blue-Green Deployment

**How It Works:**
- Two complete environments (blue = old, green = new)
- Complete switchover of user traffic in single action
- Binary deployment: users on old OR new, no overlap
- Optimized for cutover control and rollback speed
- Source: [Blue Green Deployment vs. Canary: 5 Differences](https://codefresh.io/learn/software-deployment/blue-green-deployment-vs-canary-5-key-differences-and-how-to-choose/)

**Advantages:**
- Instant rollback if issues detected
- No partial-state complexity
- Clear separation between environments

**Disadvantages:**
- Requires 2x infrastructure cost
- No gradual learning under live load
- All-or-nothing risk

**Sources:**
- [Blue-Green and Canary Deployments Explained](https://www.harness.io/blog/blue-green-canary-deployment-strategies)
- [Blue Green Deployment vs. Canary: 5 Differences & How to Choose](https://codefresh.io/learn/software-deployment/blue-green-deployment-vs-canary-5-key-differences-and-how-to-choose/)

### Canary Deployment

**How It Works:**
- Gradually roll out new version to small, controlled group
- Monitor feedback and performance
- Iterate on any issues
- Expand to larger percentage of users
- Optimized for learning under live load
- Source: [Canary vs blue-green deployment to reduce downtime](https://circleci.com/blog/canary-vs-blue-green-downtime/)

**Advantages:**
- Lower infrastructure cost (single environment)
- Gradual risk exposure
- Real-world feedback before full deployment
- Easy to expand or rollback based on metrics

**Disadvantages:**
- More complex monitoring required
- Longer deployment timeline
- Potential for inconsistent user experience during rollout

**Key Insight:** Canary optimized for learning, blue-green optimized for control. Canary better for AI/ML where real-world validation critical before full deployment.

**Sources:**
- [Canary vs blue-green deployment to reduce downtime](https://circleci.com/blog/canary-vs-blue-green-downtime/)
- [When to use canary vs. blue/green vs. rolling deployment](https://www.techtarget.com/searchitoperations/answer/When-to-use-canary-vs-blue-green-vs-rolling-deployment)
- [Deployment Strategies (Rolling, Blue-Green, Canary)](https://dev.to/godofgeeks/deployment-strategies-rolling-blue-green-canary-4ob0)

---

## Recommendations for Engram

Based on this research, here are recommended strategies for remotely updating Engram's spiking neural network brain on edge robots:

### 1. Deployment Architecture

**Shadow Mode First:**
- Deploy new belief/body-profile updates in shadow mode
- Brain processes sensory input with both old and new parameters
- Compare decision outputs
- Log divergences for analysis
- No effect on actual motor output until validated

**Canary Rollout Second:**
- After shadow mode validation, deploy to 1-5% of fleet
- Monitor for anomalies (unexpected motor patterns, safety violations, prediction error spikes)
- Gradual expansion: 5% → 10% → 25% → 50% → 100%
- Automated rollback if fleet-wide metrics degrade

### 2. Policy Switching Transient Mitigation

**Problem:** Instant parameter swap causes transient behavior (research shows no universal solution).

**Engram-Specific Approaches:**

**Option A: Smooth Interpolation (Developmental Period Model)**
- Already implemented for infant→juvenile phase transitions
- Interpolate neuromodulator baselines over N steps (e.g., 1000 steps = ~17 min at 1 Hz)
- Apply same technique to safety belief gains, body force limits, ODD boundaries
- Example: `force_limit_new = force_limit_old * (1 - alpha) + force_limit_target * alpha`, alpha from 0→1 over 1000 steps

**Option B: Eligibility Trace Gating (Architecture Invariant 1b-compatible)**
- New safety parameters update eligibility traces, not weights directly
- Neuromodulator-gated application spreads change over ~1000ms (tau of eligibility trace)
- Natural smoothing via existing learning mechanism
- No special-case code, aligns with existing architecture

**Option C: Meta-Controller Override During Transition**
- MetaControllerRegion monitors prediction error during parameter update
- If pred error spikes above threshold during transition, temporarily boost safety gains
- Self-correcting: once brain adapts to new parameters, pred error drops, safety boost removed

### 3. Rollback Mechanisms

**Weight Snapshot on Update:**
- Before applying new parameters, save to SQLite: `safety_snapshot_YYYY-MM-DD-HHMMSS`
- Include: beliefs gains, body profile, ODD bounds, neuromod baselines
- NOT full weight snapshot (too large at 1M neurons) — only changed parameters

**Automated Rollback Triggers:**
- Sustained prediction error >2σ above baseline for >60s
- Motor cortex cognitive sub-range firing rate >80% (brain confused, queries LLM constantly)
- Safety supervisor violations (force limit exceeded, ODD exit detected)
- Human operator dashboard "Emergency Rollback" button

**Rollback Procedure:**
1. Load previous snapshot from SQLite
2. Apply with smooth interpolation (Option A above, 100 steps for emergency)
3. Log rollback event to NATS `system.rollback` with reason
4. Dashboard alert to operator

### 4. Digital Twin Pre-Deployment Testing

**Engram Digital Twin Architecture:**

**Simulation Environment:**
- Duplicate neuromorphic service in Docker (new container `neuromorphic-twin`)
- Load production brain state (weights, neuron params, drives)
- Feed recorded sensory data from production fleet (video/audio/IMU logs)
- Apply new safety parameters to twin, run for 10K steps
- Compare twin motor outputs to production motor outputs on same sensory stream

**Pass Criteria for Production Deployment:**
- Prediction error distribution <1.2x production baseline
- No safety violations (force, ODD, collision proximity)
- Motor output divergence <20% from production (allows for improvement, blocks catastrophic change)

**Tool:** Extend `neuromorphic/tests/` with `test_twin_deployment.py` — loads recorded data, runs twin, validates outputs

### 5. Federated Learning (Future Phase)

**Fleet-Wide Belief Refinement:**
- Each robot logs: sensory input → motor output → proprioceptive feedback → prediction error
- Upload weight deltas (not full weights) for high-performing episodes (low pred error, task success)
- Cloud aggregates deltas via FedAvg
- New global belief parameters pushed to fleet via canary deployment

**Challenges for Engram:**
- Spiking network weights are ~14 GB at 1M neurons (CSR matrices) — cannot upload
- Solution: Only upload *belief gain changes* and *neuromod baseline changes* (tiny: <1 KB per robot)
- Weight changes emerge via on-robot STDP — no centralized weight training

### 6. Regulatory Compliance Path

**ISO 21434 Alignment:**
- Document cybersecurity risk for OTA belief updates (attack vector: malicious safety param injection)
- Implement NATS TLS + JWT authentication for update channel
- Log all parameter changes to immutable audit trail (append-only SQLite table)

**UNECE WP.29 R156 Alignment (if Engram targets automotive):**
- Maintain software/parameter version registry (already planned in meta-programmer)
- Assess if belief updates affect "type approval parameters" (e.g., max speed, braking distance)
- Flag updates requiring human review vs. auto-deployable

### 7. Geofencing & ODD Management

**Engram ODD Definition:**
- NOT just geographic (though include GPS bounds for outdoor robots)
- Attribute-rich conditions:
  - Sensory input stability (low novelty/change instincts)
  - Prediction error <threshold (brain understands environment)
  - Body temperature, battery level, network latency within bounds
  - Human proximity detection (slow down if humans nearby)

**Real-Time ODD Monitoring:**
- Kernel service monitors all ODD attributes every step
- Publishes to `robot.odd.status` (IN_ODD | APPROACHING_EXIT | ODD_EXIT)
- On ODD_EXIT: Motor cortex outputs scaled by safety factor (e.g., 0.5x speed), dashboard alert

**Geofence Implementation:**
- GPS-based for outdoor (Pi GPS sensor already in sensory gateway architecture)
- Vision-based for indoor (if brain sees "red boundary tape" → high novelty → instinctual gain → pred error → slow down)
- ODD boundaries adjustable via dashboard → NATS `config.odd.update`

### 8. Human Oversight Dashboard

**Real-Time Monitoring:**
- Prediction error time-series plot (detect confusion)
- Motor output rates by sub-range (detect over-reliance on cognitive LLM queries)
- Safety violation log (force, ODD, collision proximity)
- Neuromodulator levels (detect arousal spikes)
- Current belief gains, body profile, ODD bounds

**Alert System:**
- Thresholds configurable per robot
- Alert channels: dashboard notification, NATS `system.alert`, email/SMS for critical
- Root-cause analysis: When alert fires, log 60s of sensory input + motor output + brain state for post-analysis

**Kill Switch:**
- Dashboard button: "Emergency Stop" → publishes `motor.emergency_stop` to NATS
- Robot's actuator plugins subscribe, immediately halt all motors
- Brainstem arousal region monitors `motor.emergency_stop`, boosts inhibition to motor cortex (brain-level stop, not just actuator)
- Operator must acknowledge and manually restart

### 9. Update Delivery Mechanism

**NATS-Based Configuration Channel:**
- Dashboard publishes to `config.update.{robot_id}` with JSON payload:
  ```json
  {
    "version": "2026-03-15-v2",
    "mode": "shadow",  // or "canary" or "production"
    "parameters": {
      "beliefs.novelty_gain": 1.5,
      "body.force_limit_manipulation": 50.0,
      "odd.max_prediction_error": 0.8
    },
    "interpolation_steps": 1000,
    "rollback_triggers": {
      "pred_error_threshold": 1.0,
      "duration_seconds": 60
    }
  }
  ```
- Neuromorphic service subscribes, applies update with smooth interpolation
- Saves snapshot before applying
- Monitors rollback triggers, auto-reverts if criteria met

**Version Registry:**
- SQLite table `parameter_versions`: version, timestamp, parameters JSON, applied_mode (shadow/canary/production), rollback_count
- Dashboard queries registry, displays version history per robot

### 10. Testing Before Deployment

**Unit Tests (Pre-Production):**
- `tests/test_parameter_updates.py`: Validate smooth interpolation, rollback logic, snapshot save/load
- `tests/test_twin_simulation.py`: Run twin with new parameters on recorded data, validate outputs

**Integration Tests (Staging):**
- Deploy to 1 staging robot (in lab, not field)
- Run through task sequences: locomotion, manipulation, human interaction
- Human observer scores: safety, task success, smoothness
- Only promote to production if all tests pass

**Shadow Mode (Production):**
- Run for 24-48 hours on full fleet
- Analyze divergence logs
- If <5% of decisions diverge significantly, proceed to canary
- If >5%, iterate on parameters, re-test

---

## Conclusion

The autonomous vehicle and robotics industries have converged on a **multi-layered safety approach** for remote AI updates:

1. **Shadow mode** for zero-risk real-world validation
2. **Canary deployment** for gradual rollout with monitoring
3. **Automated rollback** with performance triggers
4. **Digital twins** for pre-deployment testing
5. **Regulatory compliance** (ISO 21434, UNECE WP.29) with formal update management systems
6. **Human oversight** with dashboards, alerts, and kill switches
7. **Smooth transitions** to mitigate policy-switching transients (active research area)

For Engram's spiking neural network brain:
- Leverage existing developmental period smooth interpolation for parameter updates
- Use eligibility traces (invariant 1b) for natural smoothing
- Shadow mode + canary deployment for fleet rollout
- Digital twin testing with recorded sensory data
- NATS-based update delivery with versioning and rollback
- Dashboard for human oversight with configurable alerts and emergency stop

**Next Steps:**
1. Implement shadow mode infrastructure in neuromorphic service
2. Add parameter snapshot/rollback to SQLite persistence
3. Build dashboard configuration update UI
4. Create digital twin test framework
5. Define Engram-specific ODD attributes and monitoring
6. Document update procedures for ISO 21434 compliance path
