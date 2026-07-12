"""
Chat engine — the LLM communication layer for the neuromorphic brain.

``ChatEngine`` owns both LLM backends (Ollama and OpenAI-compatible), the
brain-state context builders, and the full teleoperation turn (:meth:`converse`).
``aiohttp`` is imported lazily inside the network calls so the context builders
stay unit-testable without it.
"""

import logging
import os
import time
from typing import TYPE_CHECKING

from dashboard.state import DashboardState

if TYPE_CHECKING:  # avoid an import cycle at runtime
    from dashboard.nats_stream import NatsStreamManager


class ChatEngine:
    """LLM interface to the brain. Holds backend config and builds context."""

    def __init__(self, state: DashboardState, nats: "NatsStreamManager"):
        self._state = state
        self._nats = nats
        self.logger = logging.getLogger("dashboard.chat")
        self._ollama_url = os.environ.get("OLLAMA_URL", "http://ollama:11434")
        self._openai_url = os.environ.get("OPENAI_API_URL", "")
        self._openai_key = os.environ.get("OPENAI_API_KEY", "")
        self.llm_model = os.environ.get("LLM_MODEL", "llama3.2")

    # ── teleoperation turn ────────────────────────────────────────────────
    async def converse(self, user_message: str) -> dict[str, str]:
        """Run one full teleoperation turn for ``user_message``.

        Injects the text into the brain as sensory input, queries the LLM
        window for a reply, records the interaction (skill timing, flywheel,
        chat history), and returns the reply dict (``{"content", "model"}``).
        Shared by both the ``POST /api/chat`` route and the WS chat channel.
        """
        await self._nats.publish_text_observation(user_message)

        t0 = time.time()
        reply = await self._complete(user_message)
        dur = (time.time() - t0) * 1000
        self._state.skills.record_call("brain.chat", dur, reply.get("model") != "error")

        # Flywheel: every chat interaction is learning.
        self._state.knowledge.learn("teleoperation", "conversation", user_message[:200])

        self._state.append_chat("user", user_message)
        self._state.append_chat("assistant", reply["content"])
        return reply

    async def _complete(self, user_message: str) -> dict[str, str]:
        """Build the prompt from brain state + history and query the backend."""
        system_context = self._build_system_context()
        messages = [{"role": "system", "content": system_context}]
        for entry in self._state.chat_history[-20:]:
            messages.append({"role": entry["role"], "content": entry["content"]})
        messages.append({"role": "user", "content": user_message})

        if self._openai_url and self._openai_key:
            return await self._chat_openai(messages)
        return await self._chat_ollama(messages)

    # ── context builders ──────────────────────────────────────────────────
    def _build_system_context(self) -> str:
        parts = [
            "You are the voice of Engram -- a neuromorphic brain built on spiking neural networks.",
            "You are NOT the brain itself. You are a communication interface (LLM) that translates",
            "the brain's state into natural language so humans can understand what it is experiencing.",
            "",
            "Engram is a biologically-grounded cognitive architecture for robotics:",
            "- 1M+ spiking neurons with STDP learning, eligibility traces, BCM metaplasticity",
            "- 4-channel neuromodulation (dopamine, acetylcholine, norepinephrine, serotonin)",
            "- Developmental phases: infant, toddler, juvenile, adolescent, mature",
            "- Learns through sensory experience (video, audio, proprioception), NOT from text prompts",
            "- Virtual body (MuJoCo humanoid) with closed-loop motor learning",
            "- Cross-modal binding: temporal correlation between what it sees and hears forms associations via STDP",
            "",
            "IMPORTANT RULES:",
            "- Do NOT claim to notice system metrics, load averages, or performance changes yourself.",
            "  You only know what is provided in the brain state below.",
            "- Do NOT offer unsolicited maintenance advice or system health commentary.",
            "- Do NOT pretend the user's text input is directly training you. Text goes to the brain",
            "  as sensory input; your LLM response is separate.",
            "- When discussing what the brain 'knows' or 'feels', be honest about its current stage.",
            "  A juvenile brain has basic feature representations, not rich semantic understanding.",
            "- Be concise, scientifically grounded, and avoid anthropomorphizing beyond what the data shows.",
            "",
        ]

        # Brain state from neuromorphic network (real data, not hallucinated)
        brain_section = self._interpret_brain_state()
        if brain_section:
            parts.append("CURRENT BRAIN STATE (real-time data from the spiking network):")
            parts.append(brain_section)
            parts.append("")

        system_info = self._state.system_info
        if system_info:
            os_i = system_info.get("os", {})
            cpu = system_info.get("cpu", {})
            mem = system_info.get("memory", {})
            parts.append(
                f"Server: {os_i.get('system', '?')} {os_i.get('release', '')} -- {cpu.get('model', cpu.get('architecture', '?'))} ({cpu.get('cores', '?')} cores), {mem.get('total_gb', '?')} GB RAM"
            )

        parts.extend(
            [
                "",
                "When the user teaches you something (e.g., 'a ball is round'), explain that the text has",
                "been injected into the brain's sensory cortex as a spike pattern. The brain will form",
                "associations through STDP if this input correlates with other sensory experience.",
                "The brain learns from temporal correlation, not from understanding the sentence.",
            ]
        )
        return "\n".join(parts)

    def _interpret_brain_state(self) -> str:
        """Translate raw neuromorphic metrics into natural language for the LLM system prompt."""
        neuro_metrics = self._state.neuro_metrics
        if not neuro_metrics:
            return ""

        parts = []

        # Development stage -- use actual phase from neuromodulation if available,
        # otherwise estimate from step count using the real phase boundaries.
        step_count = neuro_metrics.get("step_count", 0)
        phase = neuro_metrics.get("phase", "")
        if not phase:
            if step_count < 60_000:
                phase = "infant"
            elif step_count < 360_000:
                phase = "toddler"
            elif step_count < 2_160_000:
                phase = "juvenile"
            else:
                phase = "adolescent or mature"

        phase_descriptions = {
            "infant": "infant (basic sensory calibration, high plasticity)",
            "toddler": "toddler (forming first associations, neuromodulator baselines shifting)",
            "juvenile": "juvenile (active learning, cross-modal binding forming, pre-adolescent)",
            "adolescent": "adolescent (pruning weak synapses, myelinating strong ones, locking structure)",
            "mature": "mature (stable representations, reduced plasticity, continual learning mode)",
        }
        stage = phase_descriptions.get(phase, phase)
        parts.append(f"Phase: {stage} -- step {step_count:,}")

        # Cognitive state from firing rates
        firing = neuro_metrics.get("firing_rates", {})
        cognitive_notes = []

        assoc_rate = firing.get("association_cortex", 0)
        if assoc_rate > 0.05:
            cognitive_notes.append("association cortex highly active (creative/connecting)")
        elif assoc_rate > 0.01:
            cognitive_notes.append("association cortex engaged (linking concepts)")

        pred_rate = firing.get("predictive_layer", 0)
        if pred_rate > 0.05:
            cognitive_notes.append("predictive layer firing strongly (anticipating)")
        elif pred_rate > 0.01:
            cognitive_notes.append("predictive layer active (forming expectations)")

        motor_rate = firing.get("motor_cortex", 0)
        if motor_rate > 0.05:
            cognitive_notes.append("motor cortex active (action-oriented)")

        sensory_rate = firing.get("sensory_cortex", 0)
        if sensory_rate > 0.05:
            cognitive_notes.append("sensory cortex processing input (attentive)")

        wm_rate = firing.get("working_memory", 0)
        if wm_rate > 0.03:
            cognitive_notes.append("working memory engaged (holding context)")

        if cognitive_notes:
            parts.append(f"Cognitive: {'; '.join(cognitive_notes)}")
        else:
            parts.append("Cognitive: resting state (minimal activity)")

        # Emotional state from drives
        drives = neuro_metrics.get("drives", {})
        emotional_notes = []

        energy = drives.get("energy", 1.0)
        if energy > 0.7:
            emotional_notes.append("alert and energetic")
        elif energy > 0.3:
            emotional_notes.append("moderate energy")
        else:
            emotional_notes.append("low energy (conserving)")

        fatigue = drives.get("fatigue", 0.0)
        if fatigue > 0.7:
            emotional_notes.append("fatigued (prefer brevity)")
        elif fatigue > 0.4:
            emotional_notes.append("slightly tired")

        damage = drives.get("damage", 0.0)
        if damage > 0.3:
            emotional_notes.append("sensing strain (concerned)")

        if emotional_notes:
            parts.append(f"Emotional: {', '.join(emotional_notes)}")

        # Prediction error
        pred_error = neuro_metrics.get("drives", {}).get("prediction_error")
        if pred_error is not None and pred_error > 0.5:
            parts.append(
                f"Surprise level: high ({pred_error:.2f}) — world model is being challenged"
            )

        return "\n".join(parts)

    def _generate_brain_only_response(self) -> dict[str, str]:
        """Generate a response purely from brain state when no LLM is available."""
        neuro_metrics = self._state.neuro_metrics
        if not neuro_metrics:
            return {
                "content": "The neuromorphic brain is initializing. No metrics available yet.",
                "model": "neural-only",
            }

        step_count = neuro_metrics.get("step_count", 0)
        firing = neuro_metrics.get("firing_rates", {})
        phase = neuro_metrics.get("phase", "unknown")

        parts = [f"Engram brain at step {step_count:,} ({phase} phase)."]

        # Report active regions
        active = [(k, v) for k, v in firing.items() if v > 0.01]
        if active:
            rates = ", ".join(
                f"{k} {v*100:.0f}%" for k, v in sorted(active, key=lambda x: -x[1])[:5]
            )
            parts.append(f"Active regions: {rates}.")
        else:
            parts.append("Neurons are mostly quiet. The brain needs sensory input to activate.")

        parts.append("\n*[No LLM connected. Connect Ollama for natural language responses.]*")

        return {
            "content": " ".join(parts),
            "model": "neural-only",
        }

    # ── LLM backends ──────────────────────────────────────────────────────
    async def _chat_ollama(self, messages: list[dict]) -> dict[str, str]:
        import aiohttp

        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as session:
                async with session.post(
                    f"{self._ollama_url}/api/chat",
                    json={"model": self.llm_model, "messages": messages, "stream": False},
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return {
                            "content": data.get("message", {}).get("content", "No response."),
                            "model": f"ollama/{self.llm_model}",
                        }
                    else:
                        txt = await resp.text()
                        return {
                            "content": f"⚠️ Ollama {resp.status}. Model '{self.llm_model}' may not be pulled.\n\n`docker exec activelearning-ollama ollama pull {self.llm_model}`\n\n{txt[:200]}",
                            "model": "error",
                        }
        except aiohttp.ClientConnectorError:
            return self._generate_brain_only_response()
        except Exception as e:
            return {"content": f"⚠️ LLM error: {e}", "model": "error"}

    async def _chat_openai(self, messages: list[dict]) -> dict[str, str]:
        import aiohttp

        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session:
                async with session.post(
                    f"{self._openai_url}/v1/chat/completions",
                    json={"model": self.llm_model, "messages": messages, "max_tokens": 2000},
                    headers={
                        "Authorization": f"Bearer {self._openai_key}",
                        "Content-Type": "application/json",
                    },
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return {
                            "content": data["choices"][0]["message"]["content"],
                            "model": data.get("model", self.llm_model),
                        }
                    return {"content": f"OpenAI API error: {resp.status}", "model": "error"}
        except Exception as e:
            self.logger.warning(f"OpenAI failed, falling back to Ollama: {e}")
            return await self._chat_ollama(messages)
