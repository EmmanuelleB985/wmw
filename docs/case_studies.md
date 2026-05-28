# Case Study Templates

Templates and seed examples for the four recurring failure patterns
described in §7 (Qualitative Analyses).

## Pattern 1: Correct Answer, Invalid Transition

**Template:**
- Question: [physics question]
- Model answer: [correct]
- Trace state_0: [plausible initial state]
- Trace transition: [INVALID — e.g., force direction contradicts acceleration]
- Verifier flag: transition
- Significance: Answer accuracy overstates physical coherence.

**Seed example:**
- Question: A 5 kg block slides down a 30° frictionless incline. What is the acceleration?
- Model answer: 4.9 m/s² (correct)
- Trace says: "gravity component along incline is 4.9 m/s², block accelerates uphill"
- Verifier catches: transition effect direction disagrees with force direction
- Hidden inconsistency: correct answer + invalid transition

## Pattern 2: Rules Catch Exact Errors

**Template:**
- Question: [physics question]
- Model answer: [may be correct or incorrect]
- Trace contains: [sign error, unit error, or numeric tolerance violation]
- Rule verifier: CATCHES the error
- LLM judge: MISSES the error (treats numeric detail as approximately correct)
- Significance: Deterministic checks are high-precision for exact consistency.

**Seed example:**
- Question: What is the gauge pressure at 10m depth in water?
- Trace says: "P = ρgh = 1000 × 9.8 × 10 = 98000 Pa" but answer says "98 Pa"
- Rule verifier: catches unit_scale (answer 98 vs computed 98000, ratio > 100)
- LLM judge: misses it (doesn't verify arithmetic)

## Pattern 3: LLM Judge Catches Semantic Errors

**Template:**
- Question: [physics question]
- Model answer: [may be correct or incorrect]
- Trace contains: [wrong physical principle, wrong reaction pair, or wrong causal claim]
- Rule verifier: MISSES (no rule for this type of semantic error)
- LLM judge: CATCHES (identifies the physical reasoning flaw)
- Significance: Semantic coverage expands beyond what rules can encode.

**Seed example:**
- Question: Two blocks collide elastically. What is the final velocity of block A?
- Trace applies: "Newton's third law" as the transition rule
  (should be conservation of momentum + conservation of kinetic energy)
- Rule verifier: passes (Newton's third law contains "newton" keyword)
- LLM judge: flags transition ("Newton's third law describes force pairs, not final velocities; should use conservation laws")

## Pattern 4: Both Verifiers Miss an Error

**Template:**
- Question: [physics question]
- Model answer: [correct or incorrect]
- Trace contains: [ambiguity, unstated assumption, or domain edge case]
- Rule verifier: MISSES
- LLM judge: MISSES
- Human audit: CATCHES
- Significance: Motivates human audit; documents verifier ceiling.

**Seed example:**
- Question: A ball is thrown from a cliff. What is the time to hit the ground?
- Trace assumes: "no air resistance" (reasonable)
- Trace assumes: "ground is at the base of the cliff" (unstated — could be a ledge partway down)
- Both verifiers: pass (assumptions are internally consistent)
- Human annotator: flags ambiguity in the "ground" reference

## How to Use These Templates

1. Run the evaluation pipeline on your model
2. Filter results by hidden_inconsistency (correct answer + invalid trace)
3. Filter by verifier disagreement (rules_only_labels vs judge_only_labels)
4. Select 1-2 vivid examples per pattern
5. Fill in the template fields
6. Include in the paper's qualitative analysis section
