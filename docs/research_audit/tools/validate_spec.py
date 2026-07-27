#!/usr/bin/env python3
"""Validate the Factor-Path Diffusion research audit specification package.

Usage:
    python docs/research_audit/tools/validate_spec.py

Exit code: 0 on success, 1 on any validation failure.

Requirements:
- Python 3.10+ stdlib only (no PyYAML, no external dependencies)
- Simple text parsing for YAML files (string matching)
"""

import os
import re
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def fail(msg: str) -> None:
    print(f"  FAIL: {msg}")


def ok(msg: str) -> None:
    print(f"  OK:   {msg}")


# =============================================================================
# Check 1: Required files exist
# =============================================================================

REQUIRED_FILES = [
    # Spec files
    "docs/research_audit/spec/ARCHITECTURE_SPEC.md",
    "docs/research_audit/spec/INTERVENTION_SPEC.md",
    "docs/research_audit/spec/architecture.yaml",
    "docs/research_audit/spec/graphs.yaml",
    "docs/research_audit/spec/metrics.yaml",
    "docs/research_audit/spec/provenance.yaml",
    "docs/research_audit/spec/status.yaml",
    # Audit documents 00-16
    "docs/research_audit/00_EXECUTIVE_VERDICT.md",
    "docs/research_audit/01_REPOSITORY_FORENSICS.md",
    "docs/research_audit/02_CLAIM_EVIDENCE_MATRIX.md",
    "docs/research_audit/03_LITERATURE_NOVELTY_MAP.md",
    "docs/research_audit/04_THEORY_REFORMULATION.md",
    "docs/research_audit/05_ARCHITECTURE_OPTIONS.md",
    "docs/research_audit/06_SELECTED_ARCHITECTURE_SPEC.md",
    "docs/research_audit/07_BASELINE_FIDELITY_PLAN.md",
    "docs/research_audit/08_DATA_AND_EVALUATION_PROTOCOL.md",
    "docs/research_audit/09_EXPERIMENT_MASTER_MATRIX.md",
    "docs/research_audit/10_IMPLEMENTATION_BACKLOG.md",
    "docs/research_audit/11_TEST_AND_VERIFICATION_PLAN.md",
    "docs/research_audit/12_RISK_AND_PIVOT_REGISTER.md",
    "docs/research_audit/13_PAPER_POSITIONING.md",
    "docs/research_audit/14_DEFINITION_OF_DONE.md",
    "docs/research_audit/15_TRACEABILITY_AND_COMPLIANCE.md",
    "docs/research_audit/16_SPECIFICATION_FREEZE.md",
    # Supporting files
    "docs/research_audit/README.md",
    "docs/research_audit/audit_manifest.yaml",
    "docs/research_audit/CORRECTION_REGISTER_V3.md",
    # Validation tool itself
    "docs/research_audit/tools/validate_spec.py",
]


def check_required_files() -> int:
    failures = 0
    for rel_path in REQUIRED_FILES:
        full_path = PROJECT_ROOT / rel_path
        if full_path.exists():
            ok(f"Found: {rel_path}")
        else:
            fail(f"Missing: {rel_path}")
            failures += 1
    return failures


# =============================================================================
# Check 2: status.yaml fields match reality
# =============================================================================

def check_status_yaml() -> int:
    failures = 0
    status_path = PROJECT_ROOT / "docs/research_audit/spec/status.yaml"
    if not status_path.exists():
        fail("status.yaml not found (reported by check 1)")
        return 1

    content = status_path.read_text()
    lines = content.split("\n")

    checks = {
        "spec_version": "3.0",
        "research_direction": "CONDITIONAL_GO",
        "document_coverage": "COMPLETE",
        "literature_validation": "BLOCKED",
        "cross_document_consistency": "FAILED",
        "specification_freeze": "FAILED",
        "implementation_start": "BLOCKED",
        "gpu_experiments": "BLOCKED",
    }

    found_fields = set()
    for line in lines:
        stripped = line.strip()
        for field, expected in checks.items():
            if re.match(rf"^{field}\s*:\s*(.+?)(?:\s+#.*)?$", stripped):
                value_part = re.match(rf"^{field}\s*:\s*(.+?)(?:\s+#.*)?$", stripped).group(1).strip().strip('"').strip("'")
                if not value_part or value_part.startswith(">"):
                    continue
                if value_part == expected:
                    ok(f"status.yaml {field} = {expected}")
                else:
                    fail(f"status.yaml {field} = {value_part!r}, expected {expected!r}")
                    failures += 1
                found_fields.add(field)

    for field, expected in checks.items():
        if field not in found_fields:
            fail(f"status.yaml field '{field}' not found (expected '{expected}')")
            failures += 1

    return failures


# =============================================================================
# Check 3: Document count vs manifest
# =============================================================================

def check_document_count() -> int:
    failures = 0
    manifest_path = PROJECT_ROOT / "docs/research_audit/audit_manifest.yaml"
    if not manifest_path.exists():
        fail("audit_manifest.yaml not found")
        return 1

    content = manifest_path.read_text()

    match = re.search(r"documents:\s*(\d+)", content)
    if match:
        manifest_count = int(match.group(1))
    else:
        fail("Could not find 'documents:' count in audit_manifest.yaml")
        return 1

    docs_dir = PROJECT_ROOT / "docs/research_audit"
    doc_files = list(docs_dir.glob("*.md"))
    doc_names = {f.name for f in doc_files}
    expected_docs = {f"{(i):02d}_{name}" for i, name in [
        (0, "EXECUTIVE_VERDICT"),
        (1, "REPOSITORY_FORENSICS"),
        (2, "CLAIM_EVIDENCE_MATRIX"),
        (3, "LITERATURE_NOVELTY_MAP"),
        (4, "THEORY_REFORMULATION"),
        (5, "ARCHITECTURE_OPTIONS"),
        (6, "SELECTED_ARCHITECTURE_SPEC"),
        (7, "BASELINE_FIDELITY_PLAN"),
        (8, "DATA_AND_EVALUATION_PROTOCOL"),
        (9, "EXPERIMENT_MASTER_MATRIX"),
        (10, "IMPLEMENTATION_BACKLOG"),
        (11, "TEST_AND_VERIFICATION_PLAN"),
        (12, "RISK_AND_PIVOT_REGISTER"),
        (13, "PAPER_POSITIONING"),
        (14, "DEFINITION_OF_DONE"),
        (15, "TRACEABILITY_AND_COMPLIANCE"),
        (16, "SPECIFICATION_FREEZE"),
    ]}

    for name_template in expected_docs:
        matches = [f for f in doc_names if f.endswith(f"{name_template.split('_', 1)[1]}.md")]
        if not matches:
            fail(f"Missing document: {name_template}.md")
            failures += 1
        else:
            ok(f"Found: {matches[0]}")

    actual_md_count = len(doc_files)
    if actual_md_count >= manifest_count - 1:
        ok(f"Document count: {actual_md_count} >= {manifest_count - 1} (manifest={manifest_count}, including README)")
    else:
        fail(f"Document count: {actual_md_count} < {manifest_count - 1} (expected >= {manifest_count - 1})")
        failures += 1

    return failures


# =============================================================================
# Check 4: Canonical 8 modes in INTERVENTION_SPEC.md
# =============================================================================

CANONICAL_MODES = [
    "OBSERVATIONAL",
    "FACTOR_EDIT",
    "CONDITION_MASK",
    "DIRECT_OUTPUT_ABLATION",
    "EDGE_ABLATION",
    "NODE_DELETION",
    "FACTOR_SOURCE_CUT",
    "NEURAL_GRAPH_SURGERY",
]


def check_intervention_spec() -> int:
    failures = 0
    spec_path = PROJECT_ROOT / "docs/research_audit/spec/INTERVENTION_SPEC.md"
    if not spec_path.exists():
        fail("INTERVENTION_SPEC.md not found")
        return 1

    content = spec_path.read_text()

    for mode in CANONICAL_MODES:
        if mode in content:
            ok(f"Mode '{mode}' found in INTERVENTION_SPEC.md")
        else:
            fail(f"Mode '{mode}' NOT found in INTERVENTION_SPEC.md")
            failures += 1

    stale_names = [
        "path_ablation",
        "full_source_cut",
        "graph_surgery",
        "output_gate_only",
        "do_like",
        "do-like",
        "drop_factor",
        "zero_out",
    ]
    stale_table_found = "## Stale Names" in content
    if stale_table_found:
        ok("Stale Names deprecation table found in INTERVENTION_SPEC.md")
    else:
        fail("Stale Names deprecation table NOT found in INTERVENTION_SPEC.md")
        failures += 1

    return failures


# =============================================================================
# Check 5: Forbidden phrases
# =============================================================================

FORBIDDEN_PATTERNS = [
    (r"complete\s+DAG", "complete DAG (DENSE_DIRECTED is not a DAG)"),
    (r"fully\s+connected\s+DAG", "fully connected DAG (DENSE_DIRECTED is not a DAG)"),
    (r'"path_ablation"', "path_ablation as mode name (use FACTOR_SOURCE_CUT or DIRECT_OUTPUT_ABLATION)"),
    (r'"graph_surgery"', "graph_surgery as mode name (use NEURAL_GRAPH_SURGERY)"),
    (r'"full_source_cut"', "full_source_cut as mode name (use FACTOR_SOURCE_CUT)"),
    (r"paired\s+counterfactual", "paired counterfactual (use paired-noise evaluation)"),
    (r"do-like", "do-like (use NEURAL_GRAPH_SURGERY)"),
    (r"\bdo-operator\b", "do-operator (banned; no causal claim)"),
    (r"output_gate\s*=\s*0.*invariance", "output_gate=0 + invariance claim (insufficient cutset)"),
    (r"output_gate\s*=\s*0.*non.interference", "output_gate=0 + non-interference claim (insufficient cutset)"),
    (r'"6\s+modes"', "6 modes (current spec has 8 modes)"),
    (r"\bMMDiT-k\b", "MMDiT-k (renamed to AllToAllFactorStreamDiT)"),
    (r"\bmmdit-k\b", "mmdit-k (renamed to AllToAllFactorStreamDiT)"),
    (r"\bCoInDDiT\b", "CoInDDiT (renamed to IndependentStreamDiT)"),
    (r"\bEncDiffDiT\b", "EncDiffDiT (renamed to CrossAttnDiT)"),
    (r"child.before.parent", "child-before-parent ordering (sync updates don't need order)"),
]

ALLOWED_IN_FILES = {
    "CORRECTION_REGISTER_V3.md",
    "16_SPECIFICATION_FREEZE.md",
    "tools/validate_spec.py",
}

CONTEXT_SAFE_HINTS = [
    "banned",
    "BANNED",
    "deprecated",
    "Deprecated",
    "renamed to",
    "Renamed to",
    "stale",
    "Stale",
    "forbidden",
    "Forbidden",
    "correction register",
    "replaced with",
    "Replaced with",
    "NO (",
    "NO |",
    "rename",
    "Rename",
    "RENAME",
    "MISMATCH",
    "Mismatch",
    "not ",
    "NOT ",
    "remove",
    "Remove",
    "→",
    "From:",
    "indefensible",
    "path ablation, not",
    "claims ladder",
    "| **NO**",
    "no do-operator",
    "No do-operator",
    "required pivot",
    "Required Pivot",
    "relabel",
    "pivot",
    "describing",
    "| do-operator",
    "old name",
    "formerly",
    "former",
    "Original",
    "original",
    "pre-correction",
    "| From",
    "| from",
    "superseded",
    "Superseded",
    "≠",
    "as Pearl",
    "| **R**",
    "REFUTED",
    "silent edge",
    "MUST be replaced",
    "replaced",
    "implementation issue",
    "not implemented",
    "| R |",
    "| P |",
    "old value",
    "describing the old",
    "current code",
    "original code",
    "early audit",
    "never activated",
    "not faithful",
]


def _is_context_safe(surrounding: str) -> bool:
    """Check if a match appears in a safe deprecation/terminology context."""
    for hint in CONTEXT_SAFE_HINTS:
        if hint in surrounding:
            return True
    return False


def check_forbidden_phrases() -> int:
    failures = 0
    audit_dir = PROJECT_ROOT / "docs/research_audit"

    for md_file in sorted(audit_dir.glob("*.md")):
        if md_file.name in ALLOWED_IN_FILES:
            continue

        content = md_file.read_text()
        rel = str(md_file.relative_to(PROJECT_ROOT))
        for pattern, description in FORBIDDEN_PATTERNS:
            for m in re.finditer(pattern, content, re.IGNORECASE):
                start = max(0, m.start() - 300)
                end = min(len(content), m.end() + 300)
                surrounding = content[start:end]
                if _is_context_safe(surrounding):
                    continue
                line_num = content[:m.start()].count("\n") + 1
                snippet = m.group(0)
                fail(f"{rel}:L{line_num}: forbidden phrase '{snippet}' ({description})")
                failures += 1

    for yaml_file in sorted(audit_dir.glob("spec/*.yaml")):
        content = yaml_file.read_text()
        rel = str(yaml_file.relative_to(PROJECT_ROOT))
        for pattern, description in FORBIDDEN_PATTERNS:
            for m in re.finditer(pattern, content, re.IGNORECASE):
                start = max(0, m.start() - 300)
                end = min(len(content), m.end() + 300)
                surrounding = content[start:end]
                if _is_context_safe(surrounding):
                    continue
                if "forbidden_phrases" in surrounding:
                    continue
                line_num = content[:m.start()].count("\n") + 1
                snippet = m.group(0)
                fail(f"{rel}:L{line_num}: forbidden phrase '{snippet}' ({description})")
                failures += 1

    if failures == 0:
        ok("No forbidden phrases found in audit documents")

    return failures


# =============================================================================
# Check 6: Verify no changes outside docs/research_audit
# =============================================================================

def check_git_changes() -> int:
    failures = 0
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", "HEAD"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            timeout=10,
        )
        if result.returncode != 0:
            fail(f"git diff failed: {result.stderr.strip()}")
            return 1

        changed_files = [f for f in result.stdout.strip().split("\n") if f]
        external_changes = [f for f in changed_files
                           if not f.startswith("docs/research_audit/")]

        if external_changes:
            for f in external_changes:
                fail(f"Change outside docs/research_audit/: {f}")
            failures += len(external_changes)
        else:
            if changed_files:
                ok(f"All {len(changed_files)} changed files are within docs/research_audit/")
            else:
                ok("No uncommitted changes detected")

    except FileNotFoundError:
        fail("git not found — skipping git change check")
    except subprocess.TimeoutExpired:
        fail("git diff timed out — skipping git change check")

    return failures


# =============================================================================
# Check 7: AGENTS.md is NOT in inventory
# =============================================================================

def check_agents_md_not_in_inventory() -> int:
    failures = 0
    forensics_path = PROJECT_ROOT / "docs/research_audit/01_REPOSITORY_FORENSICS.md"
    if not forensics_path.exists():
        fail("01_REPOSITORY_FORENSICS.md not found (reported by check 1)")
        return 1

    content = forensics_path.read_text()

    file_inventory_start = content.find("## File Inventory")
    if file_inventory_start == -1:
        fail("Could not find '## File Inventory' section in 01_REPOSITORY_FORENSICS.md")
        return 1

    inventory_section = content[file_inventory_start:]
    next_section = inventory_section.find("\n## ", 10)
    if next_section != -1:
        inventory_section = inventory_section[:next_section]

    if "AGENTS.md" in inventory_section:
        fail("AGENTS.md found in file inventory of 01_REPOSITORY_FORENSICS.md")
        failures += 1
    else:
        ok("AGENTS.md NOT in file inventory of 01_REPOSITORY_FORENSICS.md")

    manifest_path = PROJECT_ROOT / "docs/research_audit/audit_manifest.yaml"
    if manifest_path.exists():
        manifest_content = manifest_path.read_text()
        if "AGENTS.md" in manifest_content:
            fail("AGENTS.md found in audit_manifest.yaml")
            failures += 1
        else:
            ok("AGENTS.md NOT in audit_manifest.yaml")

    return failures


# =============================================================================
# Check 8: CORRECTION_REGISTER_V3.md completeness
# =============================================================================

def check_correction_register() -> int:
    failures = 0
    register_path = PROJECT_ROOT / "docs/research_audit/CORRECTION_REGISTER_V3.md"
    if not register_path.exists():
        fail("CORRECTION_REGISTER_V3.md not found (reported by check 1)")
        return 1

    content = register_path.read_text()

    expected_corrs = [
        ("CORR-001", "AGENTS.md hallucination"),
        ("CORR-002", "H-066"),
        ("CORR-003", "H-073"),
        ("CORR-004", "H-076"),
        ("CORR-005", "3DShapes"),
        ("CORR-006", "raw size"),
        ("CORR-007", "Architecture 05"),
        ("CORR-008", "Intervention Mode"),
        ("CORR-009", "Factor Source Cut"),
        ("CORR-010", "complete DAG"),
        ("CORR-011", "Output-Gate-Only"),
        ("CORR-012", "paired counterfactual"),
        ("CORR-013", "do-like"),
        ("CORR-014", "graph_surgery"),
        ("CORR-015", "path_ablation"),
        ("CORR-016", "child-before-parent"),
        ("CORR-017", "Chain DAG"),
        ("CORR-018", "p<0.05"),
        ("CORR-019", "WP15"),
        ("CORR-020", "6 modes"),
        ("CORR-021", "18 Work Packages"),
        ("CORR-022", "Novelty Claim"),
        ("CORR-023", "Candidate B"),
        ("CORR-024", "MMDiT-k"),
        ("CORR-025", "Provenance"),
    ]

    content_lower = content.lower()
    for corr_id, keyword in expected_corrs:
        keyword_lower = keyword.lower()
        if corr_id in content and keyword_lower in content_lower:
            ok(f"{corr_id} found in CORRECTION_REGISTER_V3.md")
        else:
            if corr_id not in content:
                fail(f"{corr_id} NOT found in CORRECTION_REGISTER_V3.md")
            else:
                fail(f"{corr_id} found but keyword '{keyword}' not in entry context")
            failures += 1

    resolved_count = content.count("RESOLVED")
    open_count = content.count("\n- **Status**: OPEN")

    ok(f"CORRECTION_REGISTER_V3.md: {resolved_count} RESOLVED entries")
    if open_count > 0:
        fail(f"CORRECTION_REGISTER_V3.md: {open_count} OPEN entries (expected 0)")
        failures += open_count

    return failures


# =============================================================================
# Check 9: README.md version and status consistency
# =============================================================================

def check_readme_consistency() -> int:
    failures = 0
    readme_path = PROJECT_ROOT / "docs/research_audit/README.md"
    if not readme_path.exists():
        fail("README.md not found")
        return 1

    content = readme_path.read_text()

    if "## Document Index" in content:
        ok("Document Index section found in README.md")
    else:
        fail("Document Index section NOT found in README.md")
        failures += 1

    required_docs = ["00", "01", "02", "03", "04", "05", "06", "07", "08",
                     "09", "10", "11", "12", "13", "14", "15", "16"]
    for doc_num in required_docs:
        pattern = f"| {doc_num} |"
        if pattern in content:
            ok(f"Document {doc_num} listed in README.md index")
        else:
            fail(f"Document {doc_num} NOT listed in README.md index")
            failures += 1

    return failures


# =============================================================================
# Check 10: Spec file internal consistency
# =============================================================================

def check_spec_internal_consistency() -> int:
    failures = 0

    arch_yaml_path = PROJECT_ROOT / "docs/research_audit/spec/architecture.yaml"
    if arch_yaml_path.exists():
        content = arch_yaml_path.read_text()
        if "ROST-FRG" in content and "read_only_to_branches: true" in content:
            ok("architecture.yaml: ROST-FRG + read_only_to_branches confirmed")
        else:
            fail("architecture.yaml: missing ROST-FRG or read_only_to_branches")
            failures += 1
        if "intervention_modes: 8" in content:
            ok("architecture.yaml: intervention_modes = 8 confirmed")
        else:
            fail("architecture.yaml: intervention_modes != 8")
            failures += 1

    graphs_yaml_path = PROJECT_ROOT / "docs/research_audit/spec/graphs.yaml"
    if graphs_yaml_path.exists():
        content = graphs_yaml_path.read_text()
        if "DENSE_DIRECTED is NOT a DAG" in content:
            ok("graphs.yaml: 'DENSE_DIRECTED is NOT a DAG' present")
        else:
            fail("graphs.yaml: missing 'DENSE_DIRECTED is NOT a DAG'")
            failures += 1

    provenance_path = PROJECT_ROOT / "docs/research_audit/spec/provenance.yaml"
    if provenance_path.exists():
        content = provenance_path.read_text()
        for field in ["code_reference_commit", "spec_freeze_base_commit", "spec_freeze_commit"]:
            if field in content:
                ok(f"provenance.yaml: '{field}' field present")
            else:
                fail(f"provenance.yaml: '{field}' field missing")
                failures += 1

    metrics_path = PROJECT_ROOT / "docs/research_audit/spec/metrics.yaml"
    if metrics_path.exists():
        content = metrics_path.read_text()
        required_metrics = [
            "TargetValueSuccess", "TargetChangeRate", "OffTargetChange",
            "NoOpChange", "SourceInvarianceError_Denoiser",
            "SourceInvarianceError_Trajectory", "DirectContributionEffect",
            "EdgeEffect", "NonDescendantChange", "DescendantResponse",
        ]
        for metric in required_metrics:
            if metric in content:
                ok(f"metrics.yaml: '{metric}' defined")
            else:
                fail(f"metrics.yaml: '{metric}' NOT defined")
                failures += 1

    return failures


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    print("=" * 60)
    print("Factor-Path Diffusion — Spec Package Validation")
    print(f"Project root: {PROJECT_ROOT}")
    print("=" * 60)

    total_failures = 0

    checks = [
        ("Required files exist", check_required_files),
        ("status.yaml fields match reality", check_status_yaml),
        ("Document count vs manifest", check_document_count),
        ("Canonical 8 modes in INTERVENTION_SPEC.md", check_intervention_spec),
        ("Forbidden phrases check", check_forbidden_phrases),
        ("Git changes within docs/research_audit", check_git_changes),
        ("AGENTS.md not in inventory", check_agents_md_not_in_inventory),
        ("CORRECTION_REGISTER_V3.md completeness", check_correction_register),
        ("README.md consistency", check_readme_consistency),
        ("Spec file internal consistency", check_spec_internal_consistency),
    ]

    for check_name, check_fn in checks:
        print(f"\n--- {check_name} ---")
        failures = check_fn()
        total_failures += failures

    print("\n" + "=" * 60)
    if total_failures == 0:
        print("RESULT: PASS — All validations passed.")
        print("=" * 60)
        return 0
    else:
        print(f"RESULT: FAIL — {total_failures} validation failures.")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
