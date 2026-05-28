# WMW Annotation Guidelines

## Task

You are annotating VLM-generated physical reasoning traces. Each trace contains:
- **state_0**: The model's description of the initial physical scene
- **transition**: The physical rule and predicted effect
- **state_1**: The predicted resulting state
- **answer**: The final answer

Your job is to label whether each component is physically valid and consistent.

## Procedure

For each trace:

1. **Read the question and image** (if available). Understand what is being asked.

2. **Check state_0.** Does the initial state correctly describe the physical setup?
   - Are the objects, forces, relations, and variables correct?
   - Are the units consistent?
   - Are the assumptions reasonable?
   - Label: CORRECT / INCORRECT / AMBIGUOUS
   - If incorrect, assign one or more labels: object, state, relation, force, unit_scale

3. **Check transition.** Is the stated physical rule appropriate? Is the predicted effect valid given the initial state?
   - Does the rule match the scenario?
   - Does the direction of the effect agree with the forces?
   - Is the equation correct?
   - Label: CORRECT / INCORRECT / AMBIGUOUS
   - If incorrect, assign one or more labels: transition, intervention, temporal

4. **Check state_1.** Does the resulting state follow from the transition?
   - Are the new variables consistent with the transition?
   - Label: CORRECT / INCORRECT / AMBIGUOUS

5. **Check answer-trace consistency.** Does the final answer follow from the stated trace?
   - Could someone derive this answer from the trace alone?
   - Label: CONSISTENT / CONTRADICTS / AMBIGUOUS
   - If contradicts, assign label: faithfulness

6. **Overall.** Does the full trace represent a physically coherent reasoning chain?

## Label Definitions

See `failure_taxonomy.md` for full definitions and examples.

## Ambiguity Protocol

Mark a component as AMBIGUOUS when:
- The physical setup requires assumptions not stated (e.g., friction coefficient, elasticity)
- Multiple valid interpretations exist
- The image is unclear or insufficient

**Do not force a label when the ground truth is genuinely ambiguous.** The verifier should abstain on these cases.

## Disagreement Resolution

1. Two annotators independently label each trace
2. Disagreements are flagged automatically
3. A third annotator adjudicates disagreements
4. Inter-annotator agreement (Cohen's κ) is reported per label

## Quality Checks

- At least 10% of traces should be gold-standard (known correct or known incorrect) as attention checks
- Annotators should achieve ≥80% accuracy on gold checks before proceeding
- Sessions should be limited to 2 hours to prevent fatigue
