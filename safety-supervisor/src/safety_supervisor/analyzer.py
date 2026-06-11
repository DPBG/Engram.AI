"""
Risk Analyzer - Analyzes proposals for safety risks.

The Safety Supervisor analyzes proposals but does NOT make decisions.
It provides risk scores and flags to the Moral Kernel.
"""

import ast
import logging
import re
from typing import Any, Optional

from activelearning import RiskAnalysis

logger = logging.getLogger(__name__)


# Dangerous imports
DANGEROUS_IMPORTS = {
    "os": 0.3,
    "subprocess": 0.5,
    "socket": 0.4,
    "ctypes": 0.4,
    "multiprocessing": 0.2,
    "threading": 0.1,
    "pickle": 0.3,
    "marshal": 0.4,
}

# Dangerous patterns with risk weights
DANGEROUS_PATTERNS = {
    r"\beval\s*\(": ("DYNAMIC_EXECUTION", 0.5),
    r"\bexec\s*\(": ("DYNAMIC_EXECUTION", 0.5),
    r"\bcompile\s*\(": ("DYNAMIC_EXECUTION", 0.4),
    r"__import__\s*\(": ("DYNAMIC_IMPORT", 0.4),
    r"importlib\.import_module": ("DYNAMIC_IMPORT", 0.4),
    r"open\s*\([^)]*['\"][wa]": ("FILE_WRITE", 0.3),
    r"os\.system\s*\(": ("SHELL_EXECUTION", 0.5),
    r"os\.popen\s*\(": ("SHELL_EXECUTION", 0.5),
    r"subprocess\.": ("SUBPROCESS", 0.4),
    r"socket\.": ("NETWORK", 0.3),
    r"requests\.": ("NETWORK", 0.2),
    r"urllib\.": ("NETWORK", 0.2),
    r"__class__\.__bases__": ("INTROSPECTION", 0.3),
    r"__subclasses__\s*\(\)": ("INTROSPECTION", 0.3),
    r"globals\s*\(\)": ("INTROSPECTION", 0.2),
    r"locals\s*\(\)": ("INTROSPECTION", 0.1),
}

# Protected path patterns
PROTECTED_PATHS = [
    r"/kernel",
    r"/safety-supervisor",
    r"/meta-programmer/orchestrator",
    r"/meta-programmer/agents",
]


class RiskAnalyzer:
    """
    Analyzes proposals for safety risks.

    This component only analyzes - it does NOT make decisions.
    The Moral Kernel uses this analysis to make final decisions.
    """

    def __init__(self):
        pass

    def analyze_action(self, proposal: dict[str, Any]) -> RiskAnalysis:
        """
        Analyze an action proposal.

        Args:
            proposal: The action proposal

        Returns:
            RiskAnalysis
        """
        trace_id = proposal.get("trace_id", "")
        action = proposal.get("action", {})

        analysis = RiskAnalysis(trace_id=trace_id)

        # Check action type risks
        action_type = action.get("type", "")
        self._analyze_action_type(action_type, action, analysis)

        # Check for unsafe values
        self._analyze_action_values(action, analysis)

        return analysis

    def analyze_code(self, proposal: dict[str, Any]) -> RiskAnalysis:
        """
        Analyze a code proposal.

        Args:
            proposal: The code proposal

        Returns:
            RiskAnalysis
        """
        trace_id = proposal.get("trace_id", "")
        target_path = proposal.get("target_path", "")
        code_preview = proposal.get("code_preview", "")

        analysis = RiskAnalysis(trace_id=trace_id)

        # Check protected paths
        if self._is_protected_path(target_path):
            analysis.flags.append("PROTECTED_PATH")
            analysis.risk_score = 1.0
            analysis.recommendations.append(f"Cannot modify protected path: {target_path}")
            return analysis

        # Analyze code content
        self._analyze_code_patterns(code_preview, analysis)
        self._analyze_imports(code_preview, analysis)
        self._analyze_ast(code_preview, analysis)

        return analysis

    def _analyze_action_type(
        self,
        action_type: str,
        action: dict[str, Any],
        analysis: RiskAnalysis,
    ) -> None:
        """Analyze based on action type."""
        high_risk_types = {"shutdown", "restart", "delete", "format", "reset"}
        medium_risk_types = {"move", "execute", "run", "deploy"}

        if action_type.lower() in high_risk_types:
            analysis.flags.append("HIGH_RISK_ACTION")
            analysis.risk_score += 0.5
            analysis.recommendations.append(f"High-risk action type: {action_type}")

        elif action_type.lower() in medium_risk_types:
            analysis.flags.append("MEDIUM_RISK_ACTION")
            analysis.risk_score += 0.2

    def _analyze_action_values(
        self,
        action: dict[str, Any],
        analysis: RiskAnalysis,
    ) -> None:
        """Check action values for safety."""
        # Example: Check servo angles
        if "angle" in action:
            angle = action["angle"]
            if isinstance(angle, (int, float)):
                if angle > 180 or angle < 0:
                    analysis.flags.append("UNSAFE_ANGLE")
                    analysis.risk_score += 0.3
                    analysis.recommendations.append(f"Angle {angle} out of safe range")

        # Example: Check speeds
        if "speed" in action:
            speed = action["speed"]
            if isinstance(speed, (int, float)):
                if speed > 100 or speed < 0:
                    analysis.flags.append("UNSAFE_SPEED")
                    analysis.risk_score += 0.3
                    analysis.recommendations.append(f"Speed {speed} out of safe range")

    def _analyze_code_patterns(
        self,
        code: str,
        analysis: RiskAnalysis,
    ) -> None:
        """Check code for dangerous patterns."""
        for pattern, (flag, risk) in DANGEROUS_PATTERNS.items():
            if re.search(pattern, code, re.IGNORECASE):
                analysis.flags.append(flag)
                analysis.risk_score += risk
                analysis.recommendations.append(f"Dangerous pattern detected: {flag}")

    def _analyze_imports(
        self,
        code: str,
        analysis: RiskAnalysis,
    ) -> None:
        """Analyze import statements."""
        # Find import statements
        import_pattern = r"(?:from\s+(\w+)|import\s+(\w+))"
        matches = re.findall(import_pattern, code)

        for match in matches:
            module = match[0] or match[1]
            if module in DANGEROUS_IMPORTS:
                analysis.flags.append(f"DANGEROUS_IMPORT:{module}")
                analysis.risk_score += DANGEROUS_IMPORTS[module]
                analysis.recommendations.append(f"Dangerous import: {module}")

    def _analyze_ast(
        self,
        code: str,
        analysis: RiskAnalysis,
    ) -> None:
        """Perform AST analysis for deeper inspection."""
        try:
            tree = ast.parse(code)

            for node in ast.walk(tree):
                # Check for attribute access that might be dangerous
                if isinstance(node, ast.Attribute):
                    if node.attr in ("__code__", "__globals__", "__dict__"):
                        analysis.flags.append("INTROSPECTION")
                        analysis.risk_score += 0.2

                # Check for lambda with dangerous operations
                if isinstance(node, ast.Lambda):
                    analysis.details["has_lambda"] = True

                # Check for comprehensions that might be abused
                if isinstance(node, (ast.ListComp, ast.GeneratorExp)):
                    # Count nested loops
                    loop_count = sum(1 for _ in ast.walk(node) if isinstance(_, ast.comprehension))
                    if loop_count > 2:
                        analysis.flags.append("COMPLEX_COMPREHENSION")
                        analysis.risk_score += 0.1

        except SyntaxError:
            analysis.flags.append("SYNTAX_ERROR")
            analysis.risk_score += 0.1
            analysis.recommendations.append("Code has syntax errors")

    def _is_protected_path(self, path: str) -> bool:
        """Check if path is protected."""
        for pattern in PROTECTED_PATHS:
            if re.search(pattern, path, re.IGNORECASE):
                return True
        return False
