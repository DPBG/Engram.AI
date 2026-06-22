"""
Meta-Programmer Agent Team - Code generation using local LLMs.

Team Structure:
- CodeGen: Generates new code from knowledge gaps
- Refactor: Improves and refactors existing code
- Tester: Writes tests and validates code
- Deployer: Manages deployment and rollback
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _sanitize(value: object, max_len: int = 200) -> str:
    """Sanitize a hardware-sourced string for use in LLM prompts.

    Strips newlines/carriage returns and truncates to *max_len*.
    """
    return str(value).replace("\n", " ").replace("\r", " ")[:max_len]


class MetaProgrammerTeam:
    """
    Code generation team with local Ollama backend.

    Uses deepseek-coder:6.7b for code generation without external APIs.
    """

    def __init__(
        self,
        ollama_url: str,
        nats_client: Any,
        sandbox_manager: Any,
        staging_manager: Any,
        db: Any,
    ):
        self.ollama_url = ollama_url
        self.nats_client = nats_client
        self.sandbox_manager = sandbox_manager
        self.staging_manager = staging_manager
        self.db = db

        # Code generation runs through the shared SDK LLM client (one reused
        # session, central timeout/retry). Low temperature keeps code stable.
        # Imported here, not at module top, so this module stays importable in
        # the neuromorphic test env that loads it without the SDK installed.
        from activelearning.llm import LLMClient, LLMConfig

        self.model_name = "deepseek-coder:6.7b"
        self._llm = LLMClient(
            LLMConfig(
                host=ollama_url,
                model=self.model_name,
                timeout=120.0,
                options={"temperature": 0.2, "top_p": 0.9},
            )
        )
        logger.info(f"Meta-Programmer team initialized with model: {self.model_name}")

    async def generate_code(
        self,
        trace_id: str,
        description: str,
        context: dict,
    ) -> dict:
        """
        Generate code for a knowledge gap.

        Args:
            trace_id: Unique identifier
            description: Human-readable gap description
            context: Additional context (sensors, search results, etc.)

        Returns:
            dict with keys:
                - success: bool
                - target_path: where to deploy code
                - code: generated code content
                - tests: generated test content
                - error: error message if failed
        """
        try:
            logger.info(f"Generating code for: {description}")

            # Extract context
            available_sensors = context.get("available_sensors", [])
            search_results = context.get("search_results", [])

            # Determine target path based on description
            target_path = self._infer_target_path(description, context)

            # Generate code prompt
            prompt = self._build_code_prompt(
                description=description,
                available_sensors=available_sensors,
                search_results=search_results,
                context=context,
            )

            # Call Ollama for code generation
            code_content = await self._call_ollama(prompt)

            # Generate tests
            test_prompt = self._build_test_prompt(
                description=description,
                code=code_content,
            )
            test_content = await self._call_ollama(test_prompt)

            return {
                "success": True,
                "target_path": target_path,
                "code": code_content,
                "tests": test_content,
                "error": None,
            }

        except Exception as e:
            logger.error(f"Code generation failed: {e}", exc_info=True)
            return {
                "success": False,
                "target_path": "",
                "code": "",
                "tests": "",
                "error": str(e),
            }

    def _infer_target_path(self, description: str, context: dict) -> str:
        """
        Infer target deployment path from description.

        Examples:
        - "Pi camera sensor plugin" → /data/plugins/pi_camera_sensor.py
        - "Arduino servo actuator" → /data/plugins/arduino_servo_actuator.py
        - "Pick up red ball task" → /data/tasks/pick_up_red_ball/task.py
        """
        desc_lower = description.lower()

        # Plugin patterns
        if "sensor" in desc_lower or "actuator" in desc_lower:
            # Extract name from description
            name = description.lower().replace(" ", "_")
            # Remove common words
            for word in ["need", "require", "want", "plugin", "sensor", "actuator"]:
                name = name.replace(word, "")
            name = "_".join(filter(None, name.split("_")))
            return f"/data/plugins/{name}.py"

        # Task patterns
        if "task" in desc_lower or any(verb in desc_lower for verb in ["pick", "place", "move", "grab", "push", "pull"]):
            name = description.lower().replace(" ", "_")
            for word in ["task", "need", "learn", "how", "to"]:
                name = name.replace(word, "")
            name = "_".join(filter(None, name.split("_")))
            return f"/data/tasks/{name}/task.py"

        # Default to plugins
        name = description.lower().replace(" ", "_")[:50]
        return f"/data/plugins/{name}.py"

    def _build_code_prompt(
        self,
        description: str,
        available_sensors: list,
        search_results: list,
        context: dict | None = None,
    ) -> str:
        """Build prompt for code generation.

        When *context* contains ``source == "gateway_device_discovery"``,
        the prompt includes device-specific metadata so the LLM can
        generate an appropriate sensor or actuator plugin.
        """
        context = context or {}
        device_section = ""
        if context.get("source") == "gateway_device_discovery":
            plugin_type = context.get("plugin_type", "sensor")
            base_class = "SensorPlugin" if plugin_type == "sensor" else "ActuatorPlugin"
            meta = context.get("metadata", {})

            device_section = f"""
**Device Hardware Info**:
- Device type: {_sanitize(context.get('device_type', 'unknown'))}
- Device ID: {_sanitize(context.get('device_id', ''))}
- Name: {_sanitize(context.get('name', ''))}
- VID/PID: {_sanitize(meta.get('vid', 'N/A'))}/{_sanitize(meta.get('pid', 'N/A'))}
- Manufacturer: {_sanitize(meta.get('manufacturer', 'N/A'))}
- Port: {_sanitize(meta.get('port', 'N/A'))}

**Plugin Requirements**:
- Extend ``{base_class}`` from ``activelearning.plugins``
- Implement ``async def capture(self)`` that reads data from the device
- Handle connection errors and reconnection
- Call ``self.emit(data)`` to publish observations
- For actuators: publish ``motor.outcome.{{channel}}`` with proprioceptive feedback
"""

        prompt = f"""You are a code generation assistant for a robotics system.

**Task**: Generate Python code for: {description}

**Available Sensors**: {", ".join(available_sensors) if available_sensors else "None"}
{device_section}
**Context from Search**:
{self._format_search_results(search_results)}

**Requirements**:
1. Import from `activelearning` SDK
2. Use SensorPlugin or ActuatorPlugin base classes
3. Include proper error handling
4. Add docstrings
5. Follow Python best practices
6. Keep it simple and focused

**Output Format**:
- Provide ONLY the Python code
- No explanations or markdown
- Start directly with imports

Generate the code:"""
        return prompt

    def _build_test_prompt(self, description: str, code: str) -> str:
        """Build prompt for test generation."""
        prompt = f"""You are a test generation assistant.

**Task**: Generate pytest tests for this code:

```python
{code}
```

**Requirements**:
1. Use pytest and pytest-asyncio
2. Test main functionality
3. Test error cases
4. Keep tests simple (target: 5 second runtime)
5. Use mocks for hardware

**Output Format**:
- Provide ONLY the test code
- No explanations or markdown
- Start directly with imports

Generate the tests:"""
        return prompt

    def _format_search_results(self, results: list) -> str:
        """Format search results for context."""
        if not results:
            return "No prior knowledge found."

        formatted = []
        for i, result in enumerate(results[:3], 1):  # Top 3 results
            formatted.append(f"{i}. {result.get('description', 'N/A')} (confidence: {result.get('confidence', 0):.2f})")

        return "\n".join(formatted)

    async def _call_ollama(self, prompt: str) -> str:
        """
        Generate code via the shared SDK LLM client and strip any markdown.

        Raises on failure so the caller fails the gap closed.
        """
        try:
            generated_text = await self._llm.generate(prompt)
            return self._clean_generated_code(generated_text)
        except Exception as e:
            logger.error(f"Ollama API call failed: {e}")
            raise

    async def close(self) -> None:
        """Release the shared LLM HTTP session."""
        await self._llm.close()

    def _clean_generated_code(self, text: str) -> str:
        """
        Clean up generated code by removing markdown formatting.
        """
        # Remove markdown code blocks
        if "```python" in text:
            text = text.split("```python")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        # Strip whitespace
        text = text.strip()

        return text

    async def refactor_code(
        self,
        trace_id: str,
        original_code: str,
        refactor_goals: str,
    ) -> dict:
        """
        Refactor existing code.

        Args:
            trace_id: Unique identifier
            original_code: Code to refactor
            refactor_goals: What to improve

        Returns:
            dict with refactored code
        """
        # TODO: Implement refactoring pipeline
        logger.warning("Refactor not yet implemented")
        return {
            "success": False,
            "error": "Refactor agent not implemented",
        }

    async def generate_tests(
        self,
        trace_id: str,
        code: str,
    ) -> dict:
        """
        Generate tests for existing code.

        Args:
            trace_id: Unique identifier
            code: Code to test

        Returns:
            dict with test code
        """
        try:
            test_prompt = self._build_test_prompt("Generate tests", code)
            test_content = await self._call_ollama(test_prompt)

            return {
                "success": True,
                "tests": test_content,
                "error": None,
            }

        except Exception as e:
            logger.error(f"Test generation failed: {e}")
            return {
                "success": False,
                "tests": "",
                "error": str(e),
            }
