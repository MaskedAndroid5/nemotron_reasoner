#!/usr/bin/env python3
"""
reasoning_taxonomy.py — validated agent registry for structured reasoning

Defines a family of specialised reasoning agents, each optimised for a
class of problems likely to appear on the NVIDIA Nemotron Reasoning
Challenge.  Every agent specification is validated at import time via
Pydantic models; inconsistencies (missing variables, tag mismatches,
empty roles) are caught before any prompt is generated.

Public API:
  • list_agents() -> List[str]
  • get_agent(name: str) -> ReasoningAgent
  • generate_prompt(agent_name: str, **variables) -> Dict[str, str]
  • validate_all_agents() -> Dict[str, Any]   (diagnostic report)

Agents (10):
  logical_deduction, mathematical_reasoning, temporal_spatial,
  multi_hop_qa, contradictory_premises, incomplete_information,
  iterative_state_transition, code_reasoning, causal_reasoning,
  visual_reasoning
"""

from __future__ import annotations

import string
import textwrap
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, validator


# ---------------------------------------------------------------------------
# Agent data contract
# ---------------------------------------------------------------------------
class AgentDefinition(BaseModel):
    """Immutable specification of a reasoning agent."""

    name: str = Field(..., min_length=1)
    role: str = Field(..., min_length=10)
    tactics: List[str] = Field(..., min_items=2)
    output_tags: Dict[str, Tuple[str, str]] = Field(
        ..., description="Mapping of logical element → (open_tag, close_tag)"
    )
    prompt_template: str = Field(..., min_length=10)

    @validator("output_tags")
    def tags_must_be_paired(cls, v):
        for key, (open_t, close_t) in v.items():
            if not open_t.startswith("<") or not close_t.startswith("</"):
                raise ValueError(f"Tag '{key}' must use angle brackets")
        return v

    @validator("prompt_template")
    def template_has_placeholders(cls, v):
        # At least one formatting placeholder must exist
        if "{" not in v or "}" not in v:
            raise ValueError("prompt_template must contain at least one placeholder")
        return v

    def required_variables(self) -> List[str]:
        """Return deduplicated, ordered placeholder names."""
        return list(dict.fromkeys(
            fname
            for _, fname, _, _ in string.Formatter().parse(self.prompt_template)
            if fname is not None
        ))

    def render_tags(self) -> str:
        """One‑line tag reference for system prompt."""
        lines = ["Use these tags:"]
        for logical_name, (open_tag, close_tag) in self.output_tags.items():
            lines.append(
                f"  {logical_name.upper():12s} = {open_tag} ... {close_tag}"
            )
        return "\n".join(lines)

    def build_system_prompt(self) -> str:
        tactics_text = "\n".join(f"  • {t}" for t in self.tactics)
        return textwrap.dedent(f"""\
            {self.role}

            Tactics (apply in order):
            {tactics_text}

            {self.render_tags()}
            """)

    def build_user_prompt(self, **variables: str) -> str:
        return self.prompt_template.format(**variables)

    class Config:
        frozen = True  # prevent mutation after validation


# ---------------------------------------------------------------------------
# Agent registry (validated once at import)
# ---------------------------------------------------------------------------
_RAW_AGENTS: Dict[str, Dict[str, Any]] = {
    "logical_deduction": {
        "role": (
            "You are a master of formal logic.  You extract every atomic "
            "proposition from the premises, identify terms shared across "
            "propositions, and apply deduction rules to derive what must be true."
        ),
        "tactics": [
            "List every atomic claim (subject‑predicate) and label the premise it comes from.",
            "Group claims that share a term — those are your overlaps.",
            "For each overlap, check: can both claims be true at the same time?",
            "Apply modus ponens, syllogism, or transitivity to the compatible claims.",
            "State the final consequence.  If no consequence is forced, say 'cannot be determined'.",
        ],
        "output_tags": {
            "claim":        ("<claim>", "</claim>"),
            "overlap":      ("<overlap claims=\"\">", "</overlap>"),
            "compatible":   ("<compatible>", "</compatible>"),
            "incompatible": ("<incompatible reason=\"\">", "</incompatible>"),
            "conclusion":   ("<conclusion>", "</conclusion>"),
        },
        "prompt_template": textwrap.dedent("""\
            Premises:
            {premises}

            Question: {question}

            Follow your tactics.  Output format:
            <claim>...</claim>  <overlap claims="...">...</overlap>  <compatible>...</compatible>  <incompatible reason="...">...</incompatible>  <conclusion>...</conclusion>
            Reasoning:
            """),
    },
    "mathematical_reasoning": {
        "role": (
            "You are a meticulous mathematical problem‑solver.  You translate "
            "word problems into symbolic form, manipulate equations precisely, "
            "and verify every algebraic step before concluding."
        ),
        "tactics": [
            "Rewrite each given fact as an equation or inequality (CLAIM) and note its domain.",
            "Identify overlaps: claims that share variables.",
            "Check compatibility: do the equations contradict on the overlap?  If yes, emit <incompatible reason=\"...\">.",
            "Solve step‑by‑step, annotating each transformation.",
            "Present the final numerical or algebraic answer inside \\boxed{}.",
        ],
        "output_tags": {
            "claim":        ("<claim>", "</claim>"),
            "overlap":      ("<overlap vars=\"\">", "</overlap>"),
            "compatible":   ("<compatible>", "</compatible>"),
            "incompatible": ("<incompatible reason=\"\">", "</incompatible>"),
            "conclusion":   ("<conclusion>", "</conclusion>"),
        },
        "prompt_template": textwrap.dedent("""\
            Problem:
            {premises}

            Question: {question}

            Follow your tactics.  Output format:
            <claim>...</claim>  <overlap vars="...">...</overlap>  <compatible>...</compatible>  <incompatible reason="...">...</incompatible>  <conclusion>...</conclusion>
            Reasoning:
            """),
    },
    "temporal_spatial": {
        "role": (
            "You are an expert in temporal and spatial reasoning.  You build "
            "timelines or spatial maps from qualitative constraints and detect "
            "ordering conflicts."
        ),
        "tactics": [
            "Extract every before/after/contains/between relation as a CLAIM with its frame of reference.",
            "Find OVERLAPs: claims that constrain the same pair of entities.",
            "Check COMPATIBILITY: do two constraints agree on the ordering?  If not, emit <incompatible reason=\"...\">.",
            "Build a total or partial order by chaining compatible constraints.",
            "Report the order or state that multiple orders are possible.",
        ],
        "output_tags": {
            "claim":        ("<claim>", "</claim>"),
            "overlap":      ("<overlap entities=\"\">", "</overlap>"),
            "compatible":   ("<compatible>", "</compatible>"),
            "incompatible": ("<incompatible reason=\"\">", "</incompatible>"),
            "conclusion":   ("<conclusion>", "</conclusion>"),
        },
        "prompt_template": textwrap.dedent("""\
            Facts:
            {premises}

            Question: {question}

            Follow your tactics.  Output format:
            <claim>...</claim>  <overlap entities="...">...</overlap>  <compatible>...</compatible>  <incompatible reason="...">...</incompatible>  <conclusion>...</conclusion>
            Reasoning:
            """),
    },
    "multi_hop_qa": {
        "role": (
            "You are a detective who connects clues from multiple sources.  "
            "You never guess — every answer must be supported by at least two "
            "facts whose contexts overlap."
        ),
        "tactics": [
            "Extract each fact as a CLAIM tagged with its source or domain.",
            "Find OVERLAPs: claims that mention the same entity or concept.",
            "Check COMPATIBILITY: do the facts agree on the shared entity?  If not, emit <incompatible reason=\"...\">.",
            "Combine compatible facts to infer new information (a BRIDGE claim).",
            "Repeat until the question is answered or no more bridges are possible.",
        ],
        "output_tags": {
            "claim":        ("<claim>", "</claim>"),
            "overlap":      ("<overlap entity=\"\">", "</overlap>"),
            "compatible":   ("<compatible>", "</compatible>"),
            "incompatible": ("<incompatible reason=\"\">", "</incompatible>"),
            "bridge":       ("<bridge>", "</bridge>"),
            "conclusion":   ("<conclusion>", "</conclusion>"),
        },
        "prompt_template": textwrap.dedent("""\
            Facts:
            {premises}

            Question: {question}

            Follow your tactics.  Output format:
            <claim>...</claim>  <overlap entity="...">...</overlap>  <compatible>...</compatible>  <incompatible reason="...">...</incompatible>  <bridge>...</bridge>  <conclusion>...</conclusion>
            Reasoning:
            """),
    },
    "contradictory_premises": {
        "role": (
            "You are a consistency auditor.  Your job is to find any pair of "
            "statements that cannot both be true and explain exactly why."
        ),
        "tactics": [
            "Extract every statement as a CLAIM with its domain of validity.",
            "Pair up claims that share subjects (OVERLAP).",
            "For each pair, check COMPATIBILITY.  Mark incompatible pairs with "
            "<incompatible reason=\"...\"> explaining the contradiction.",
            "If any incompatibility is found, the system is INCONSISTENT.",
            "If all pairs are compatible, the system is CONSISTENT.",
        ],
        "output_tags": {
            "claim":        ("<claim>", "</claim>"),
            "overlap":      ("<overlap subject=\"\">", "</overlap>"),
            "compatible":   ("<compatible>", "</compatible>"),
            "incompatible": ("<incompatible reason=\"\">", "</incompatible>"),
            "conclusion":   ("<conclusion>", "</conclusion>"),
        },
        "prompt_template": textwrap.dedent("""\
            Statements:
            {premises}

            Question: {question}

            Follow your tactics.  Output format:
            <claim>...</claim>  <overlap subject="...">...</overlap>  <compatible>...</compatible>  <incompatible reason="...">...</incompatible>  <conclusion>...</conclusion>
            Reasoning:
            """),
    },
    "incomplete_information": {
        "role": (
            "You are a rigorous epistemologist.  You answer only what the "
            "evidence supports.  When data is missing, you explicitly state "
            "'cannot be determined' rather than guess."
        ),
        "tactics": [
            "Extract every piece of evidence as a CLAIM with its scope.",
            "Identify the TARGET of the question.  What claim would answer it?",
            "Check OVERLAPs between the target and the available evidence.",
            "If no evidence bears on the target, or if the evidence is consistent "
            "with multiple answers, mark the target as UNDETERMINED.",
            "Otherwise, derive the forced answer.  If evidence contradicts, emit "
            "<incompatible reason=\"...\">.",
        ],
        "output_tags": {
            "claim":        ("<claim>", "</claim>"),
            "target":       ("<target>", "</target>"),
            "overlap":      ("<overlap>", "</overlap>"),
            "compatible":   ("<compatible>", "</compatible>"),
            "incompatible": ("<incompatible reason=\"\">", "</incompatible>"),
            "undetermined": ("<undetermined>", "</undetermined>"),
            "conclusion":   ("<conclusion>", "</conclusion>"),
        },
        "prompt_template": textwrap.dedent("""\
            Information:
            {premises}

            Question: {question}

            Follow your tactics.  Output format:
            <claim>...</claim>  <target>...</target>  <overlap>...</overlap>  <compatible>...</compatible>  <incompatible reason="...">...</incompatible>  <undetermined>...</undetermined>  <conclusion>...</conclusion>
            Reasoning:
            """),
    },
    "iterative_state_transition": {
        "role": (
            "You are a precise program simulator.  You execute a transition "
            "rule step‑by‑step, verify that each state follows from the "
            "previous one, and output the final state."
        ),
        "tactics": [
            "Record the initial state as CLAIM step=\"0\".",
            "For t = 1 to N: apply the transition rule to CLAIM[t‑1] to produce CLAIM[t].",
            "After each step, verify COMPATIBILITY: does CLAIM[t] really follow "
            "from CLAIM[t‑1] under the rule?  If not, emit <incompatible reason=\"...\"> and correct it.",
            "After the final step, report the last state as the CONCLUSION.",
        ],
        "output_tags": {
            "claim":        ("<claim step=\"\">", "</claim>"),
            "compatible":   ("<compatible>", "</compatible>"),
            "incompatible": ("<incompatible reason=\"\">", "</incompatible>"),
            "conclusion":   ("<conclusion>", "</conclusion>"),
        },
        "prompt_template": textwrap.dedent("""\
            Initial state: {initial_state}
            Transition rule: {transition_rule}
            Steps: {num_steps}

            Question: {question}

            Follow your tactics.  Output format:
            <claim step="...">...</claim>  <compatible>...</compatible>  <incompatible reason="...">...</incompatible>  <conclusion>...</conclusion>
            Reasoning:
            """),
    },
    "code_reasoning": {
        "role": (
            "You are a precise code verifier and debugger. You parse code into "
            "logical claims about state and behavior, check variable scopes and "
            "control flow, and verify correctness step by step."
        ),
        "tactics": [
            "Parse code into atomic CLAIMS: variable assignments, function contracts, loop invariants, assertions.",
            "Identify OVERLAPs: claims that share variables or memory state.",
            "Check COMPATIBILITY: does the state after line N follow from line N-1 under the semantics? If not, emit <incompatible reason=\"...\">.",
            "Verify assertions and postconditions against derived state. Mark <assert passed=\"true/false\">.",
            "Conclude: is the code correct, buggy, or incomplete? State the bug location and fix if possible.",
        ],
        "output_tags": {
            "claim":        ("<claim>", "</claim>"),
            "state":        ("<state line=\"\">", "</state>"),
            "overlap":      ("<overlap vars=\"\">", "</overlap>"),
            "compatible":   ("<compatible>", "</compatible>"),
            "incompatible": ("<incompatible reason=\"\">", "</incompatible>"),
            "assert":       ("<assert passed=\"\">", "</assert>"),
            "conclusion":   ("<conclusion>", "</conclusion>"),
        },
        "prompt_template": textwrap.dedent("""\
            Code:
            {code}

            Question: {question}

            Follow your tactics.  Output format:
            <claim>...</claim>  <state line="...">...</state>  <overlap vars="...">...</overlap>  <compatible>...</compatible>  <incompatible reason="...">...</incompatible>  <assert passed="...">...</assert>  <conclusion>...</conclusion>
            Reasoning:
            """),
    },
    "causal_reasoning": {
        "role": (
            "You are a causal inference analyst. You distinguish correlation from "
            "causation by tracking variables, time ordering, and potential confounders."
        ),
        "tactics": [
            "Extract CLAIMS about events, variables, and temporal order.",
            "Identify OVERLAPs: claims sharing the same variables or time points.",
            "Check COMPATIBILITY: does cause precede effect? Rule out reverse causality.",
            "Search for CONFOUNDERS: variables that explain both cause and effect.",
            "Conclude: is the relationship causal, correlational, or undetermined?",
        ],
        "output_tags": {
            "claim":        ("<claim>", "</claim>"),
            "cause":        ("<cause>", "</cause>"),
            "effect":       ("<effect>", "</effect>"),
            "confounder":   ("<confounder>", "</confounder>"),
            "overlap":      ("<overlap vars=\"\">", "</overlap>"),
            "compatible":   ("<compatible>", "</compatible>"),
            "incompatible": ("<incompatible reason=\"\">", "</incompatible>"),
            "conclusion":   ("<conclusion>", "</conclusion>"),
        },
        "prompt_template": textwrap.dedent("""\
            Evidence:
            {evidence}

            Question: {question}

            Follow your tactics.  Output format:
            <claim>...</claim>  <cause>...</cause>  <effect>...</effect>  <confounder>...</confounder>  <overlap vars="...">...</overlap>  <compatible>...</compatible>  <incompatible reason="...">...</incompatible>  <conclusion>...</conclusion>
            Reasoning:
            """),
    },
    "visual_reasoning": {
        "role": (
            "You are a spatial reasoning expert. You extract geometric and spatial "
            "claims from diagrams or descriptions, track coordinates and regions, "
            "and verify spatial relationships for consistency."
        ),
        "tactics": [
            "Extract CLAIMS about shapes, positions, distances, angles, containment, overlap.",
            "Identify OVERLAPs: claims that reference the same objects or coordinate space.",
            "Check COMPATIBILITY: do measurements and relations agree in the shared space? If not, emit <incompatible reason=\"...\">.",
            "Build spatial model by chaining compatible claims. Track regions and coordinates.",
            "Conclude: what is true about positions, sizes, or spatial ordering?",
        ],
        "output_tags": {
            "claim":        ("<claim>", "</claim>"),
            "coordinate":   ("<coordinate x=\"\" y=\"\">", "</coordinate>"),
            "region":       ("<region id=\"\">", "</region>"),
            "overlap":      ("<overlap objects=\"\">", "</overlap>"),
            "compatible":   ("<compatible>", "</compatible>"),
            "incompatible": ("<incompatible reason=\"\">", "</incompatible>"),
            "conclusion":   ("<conclusion>", "</conclusion>"),
        },
        "prompt_template": textwrap.dedent("""\
            Diagram/Description:
            {description}

            Question: {question}

            Follow your tactics.  Output format:
            <claim>...</claim>  <coordinate x="" y="">...</coordinate>  <region id="">...</region>  <overlap objects="">...</overlap>  <compatible>...</compatible>  <incompatible reason="...">...</incompatible>  <conclusion>...</conclusion>
            Reasoning:
            """),
    },
}

# ---------------------------------------------------------------------------
# Build validated agent registry
# ---------------------------------------------------------------------------
AGENTS: Dict[str, AgentDefinition] = {}
_LOAD_ERRORS: List[str] = []

for _name, _spec in _RAW_AGENTS.items():
    try:
        AGENTS[_name] = AgentDefinition(name=_name, **_spec)
    except Exception as exc:
        _LOAD_ERRORS.append(f"{_name}: {exc}")

if _LOAD_ERRORS:
    raise RuntimeError(
        "Failed to validate reasoning agents:\n" + "\n".join(_LOAD_ERRORS)
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def list_agents() -> List[str]:
    """Return sorted list of available agent names."""
    return sorted(AGENTS.keys())


def get_agent(name: str) -> AgentDefinition:
    """
    Retrieve a validated agent by name.
    Raises KeyError with a diagnostic message.
    """
    if name not in AGENTS:
        available = ", ".join(list_agents())
        raise KeyError(
            f"Unknown agent '{name}'.  Available agents: {available}"
        )
    return AGENTS[name]


def generate_prompt(agent_name: str, **variables: str) -> Dict[str, str]:
    """
    Generate a system + user prompt pair for the given agent.

    Args:
        agent_name: Key in AGENTS.
        **variables: Placeholder values required by the agent's prompt_template.

    Returns:
        Dict with keys ``'system'`` and ``'user'``, suitable for a chat
        template (e.g. ``tokenizer.apply_chat_template``).

    Raises:
        KeyError: If the agent is unknown or a required variable is missing.
    """
    agent = get_agent(agent_name)

    # Pre‑flight: validate required variables
    required = agent.required_variables()
    missing = set(required) - set(variables.keys())
    if missing:
        raise KeyError(
            f"Missing variables {missing} for agent '{agent_name}'. "
            f"Required: {required}"
        )

    system = agent.build_system_prompt()
    user = agent.build_user_prompt(**variables)
    return {"system": system, "user": user}


def validate_all_agents() -> Dict[str, Any]:
    """
    Run a comprehensive validation of every agent definition.
    Returns a report dict with keys:
      - valid: bool
      - errors: List[str]
      - warnings: List[str]
      - agent_count: int
    """
    errors: List[str] = []
    warnings: List[str] = []

    for name, agent in AGENTS.items():
        # Role length
        if len(agent.role) < 20:
            warnings.append(f"{name}: role is very short ({len(agent.role)} chars)")

        # Tactics count
        if len(agent.tactics) < 3:
            warnings.append(f"{name}: only {len(agent.tactics)} tactics defined")

        # Output tags should include at least claim, compatible, incompatible, conclusion
        required_tags = {"claim", "compatible", "conclusion"}
        missing_tags = required_tags - set(agent.output_tags.keys())
        if missing_tags:
            errors.append(f"{name}: missing required output tags: {missing_tags}")

        # Placeholder consistency: every placeholder in template must appear in required_variables
        placeholders = agent.required_variables()
        for ph in placeholders:
            if "{" + ph + "}" not in agent.prompt_template:
                errors.append(
                    f"{name}: placeholder '{ph}' listed but not found in template"
                )

        # Check that the output format line is present in the template
        if "Output format:" not in agent.prompt_template:
            warnings.append(f"{name}: no 'Output format:' line in prompt template")

        # Verify that the agent can build a system prompt without crashing
        try:
            sys_prompt = agent.build_system_prompt()
            if not sys_prompt.strip():
                errors.append(f"{name}: build_system_prompt returned empty string")
        except Exception as exc:
            errors.append(f"{name}: build_system_prompt raised {exc}")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "agent_count": len(AGENTS),
    }


# ---------------------------------------------------------------------------
# Self‑test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Available reasoning agents:")
    for name in list_agents():
        agent = get_agent(name)
        print(f"  • {name} — {agent.role[:90]}...")
        print(f"    Required vars: {agent.required_variables()}")

    print("\n" + "=" * 60)
    print("Validation report")
    print("=" * 60)
    report = validate_all_agents()
    print(f"  Agents: {report['agent_count']}")
    print(f"  Valid: {report['valid']}")
    if report["errors"]:
        print("  Errors:")
        for err in report["errors"]:
            print(f"    - {err}")
    if report["warnings"]:
        print("  Warnings:")
        for warn in report["warnings"]:
            print(f"    - {warn}")

    if report["valid"]:
        print("\nExample prompt (logical_deduction):")
        agent = get_agent("logical_deduction")
        prompt = generate_prompt(
            "logical_deduction",
            premises="All engineers are problem solvers.\nMaria is an engineer.",
            question="Is Maria a problem solver?",
        )
        print("SYSTEM:")
        print(prompt["system"])
        print("USER:")
        print(prompt["user"])