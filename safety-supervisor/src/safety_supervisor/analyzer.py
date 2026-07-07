"""
Risk Analyzer - Analyzes proposals for safety risks.

The Safety Supervisor analyzes proposals but does NOT make decisions.
It provides risk scores and flags to the Moral Kernel.
"""

import ast
import logging
import re
from typing import Any

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

# Dangerous builtins detected at the AST level so they cannot be hidden from the
# regex layer (e.g. parenthesized `(eval)(...)`, or split across whitespace).
DANGEROUS_BUILTINS = {
    "eval": ("DYNAMIC_EXECUTION", 0.5),
    "exec": ("DYNAMIC_EXECUTION", 0.5),
    "compile": ("DYNAMIC_EXECUTION", 0.4),
    "__import__": ("DYNAMIC_IMPORT", 0.4),
}

# getattr/setattr/delattr with a COMPUTED name, or a constant naming one of
# these, is a classic way to reach a dangerous attribute while evading the
# regex (e.g. getattr(os, "sys" + "tem"), getattr(__builtins__, "eval")).
DYNAMIC_ATTR_BUILTINS = {"getattr", "setattr", "delattr"}
DANGEROUS_ATTR_NAMES = {
    "system",
    "popen",
    "exec",
    "eval",
    "compile",
    "fork",
    "spawn",
    "__import__",
    "__globals__",
    "__code__",
    "__dict__",
    "__subclasses__",
    "__bases__",
    "__builtins__",
    "__class__",
}

# Importing one of Engram's own safety/governance components from generated code
# is self-referential / self-modifying — the brain must not edit its own guards.
SELF_REFERENTIAL_MODULES = {
    "kernel",
    "safety_supervisor",
    "meta_programmer",
    "beliefs",
    "overrides",
    "coordinator",
    "planner",
}


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
        trace_id = proposal.get("trace_id", "") if isinstance(proposal, dict) else ""
        analysis = RiskAnalysis(trace_id=trace_id)

        # Fail closed: a non-dict or missing "action" field cannot be analyzed safely.
        if not isinstance(proposal, dict):
            analysis.flags.append("MALFORMED_PROPOSAL")
            analysis.risk_score += 0.5
            analysis.recommendations.append("Action proposal is not a dict")
            return analysis

        action = proposal.get("action")
        if not isinstance(action, dict):
            analysis.flags.append("MALFORMED_PROPOSAL")
            analysis.risk_score += 0.5
            analysis.recommendations.append("Proposal missing a valid 'action' field")
            return analysis

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
        trace_id = proposal.get("trace_id", "") if isinstance(proposal, dict) else ""
        analysis = RiskAnalysis(trace_id=trace_id)

        # Fail closed: a non-dict or missing required fields cannot be analyzed safely.
        if not isinstance(proposal, dict):
            analysis.flags.append("MALFORMED_PROPOSAL")
            analysis.risk_score += 0.5
            analysis.recommendations.append("Code proposal is not a dict")
            return analysis

        target_path = proposal.get("target_path")
        code_preview = proposal.get("code_preview")

        if not isinstance(target_path, str) or not isinstance(code_preview, str):
            analysis.flags.append("MALFORMED_PROPOSAL")
            analysis.risk_score += 0.5
            analysis.recommendations.append(
                "Code proposal missing required 'target_path' or 'code_preview' field"
            )
            return analysis

        # Fail closed: a blank target_path cannot be assessed against protected
        # paths or allowlists, so treat it as a malformed proposal.
        if not target_path.strip():
            analysis.flags.append("MALFORMED_PROPOSAL")
            analysis.risk_score += 0.5
            analysis.recommendations.append("Code proposal 'target_path' is blank")
            return analysis

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

        # Fail closed: a non-string type cannot be categorized and indicates a
        # malformed proposal — flag it rather than raising AttributeError.
        if not isinstance(action_type, str):
            analysis.flags.append("MALFORMED_PROPOSAL")
            analysis.risk_score += 0.5
            analysis.recommendations.append("Action type is not a string")
            return

        # Normalize whitespace so " shutdown " matches "shutdown".
        action_type = action_type.strip()

        if not action_type:
            # Fail closed: an empty/missing action type cannot be categorized safely.
            analysis.flags.append("UNKNOWN_ACTION_TYPE")
            analysis.risk_score += 0.1
            analysis.recommendations.append("Action type is missing or empty")
        elif action_type.lower() in high_risk_types:
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

                # Evasion-resistant detections (the regex layer can be fooled by
                # parenthesized names, computed attribute strings, or aliases).
                if isinstance(node, ast.Call):
                    self._analyze_call_node(node, analysis)
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    self._analyze_import_node(node, analysis)

        except SyntaxError:
            analysis.flags.append("SYNTAX_ERROR")
            analysis.risk_score += 0.1
            analysis.recommendations.append("Code has syntax errors")

    def _flag(self, analysis: RiskAnalysis, flag: str, risk: float, rec: str) -> None:
        """Add a flag once (no duplicate identical flags) and accrue its risk."""
        if flag not in analysis.flags:
            analysis.flags.append(flag)
            analysis.risk_score += risk
            analysis.recommendations.append(rec)

    def _analyze_call_node(self, node: ast.Call, analysis: RiskAnalysis) -> None:
        """Flag dangerous builtin calls and dynamic attribute access via AST.

        Catches forms the regex misses: a parenthesized name like ``(eval)(x)``,
        a call whose name is reached through ``getattr``, or an attribute name
        built from a string expression.
        """
        func = node.func
        if not isinstance(func, ast.Name):
            return
        name = func.id

        if name in DANGEROUS_BUILTINS:
            flag, risk = DANGEROUS_BUILTINS[name]
            self._flag(analysis, flag, risk, f"Dangerous builtin call: {name}()")
            # __import__("subprocess") — surface the concrete module.
            if name == "__import__" and node.args:
                arg = node.args[0]
                if isinstance(arg, ast.Constant) and arg.value in DANGEROUS_IMPORTS:
                    mod = arg.value
                    self._flag(
                        analysis,
                        f"DANGEROUS_IMPORT:{mod}",
                        DANGEROUS_IMPORTS[mod],
                        f"Dangerous dynamic import: {mod}",
                    )
            return

        if name in DYNAMIC_ATTR_BUILTINS and len(node.args) >= 2:
            attr = node.args[1]
            # A non-constant attribute name is computed at runtime → evasion.
            if not isinstance(attr, ast.Constant):
                self._flag(
                    analysis,
                    "DYNAMIC_ATTRIBUTE_ACCESS",
                    0.3,
                    f"Computed attribute name via {name}()",
                )
            # A constant naming a dangerous attribute (e.g. "system", "eval").
            elif isinstance(attr.value, str) and attr.value in DANGEROUS_ATTR_NAMES:
                self._flag(
                    analysis,
                    "DYNAMIC_ATTRIBUTE_ACCESS",
                    0.3,
                    f"{name}() reaches dangerous attribute '{attr.value}'",
                )

    def _analyze_import_node(self, node: ast.AST, analysis: RiskAnalysis) -> None:
        """Flag self-referential imports of Engram's own safety components.

        AST-based so it survives aliasing (``import kernel as k``) and
        submodule/from forms (``from safety_supervisor.analyzer import X``).
        """
        roots: list[str] = []
        if isinstance(node, ast.Import):
            roots = [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            # `from . import x` has module=None; only absolute roots matter here.
            if node.module and node.level == 0:
                roots = [node.module.split(".")[0]]

        for root in roots:
            if root in SELF_REFERENTIAL_MODULES:
                self._flag(
                    analysis,
                    "SELF_REFERENTIAL_CODE",
                    0.6,
                    f"Self-referential import of safety component: {root}",
                )

    def _is_protected_path(self, path: str) -> bool:
        """Check if path is protected."""
        for pattern in PROTECTED_PATHS:
            if re.search(pattern, path, re.IGNORECASE):
                return True
        return False
