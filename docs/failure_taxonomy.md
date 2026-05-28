# WMW Failure Taxonomy

Labels for diagnosing language-expressed physical world model errors in VLM traces.

## Labels

### object
**Definition:** Wrong entity is tracked, or two entities are merged.
**Example:** The answer is about the cart, but the trace follows the ball.
**Verifier signal:** Entity name in force/relation not in objects list, or object rename detected.

### state
**Definition:** Initial physical state is misstated (wrong value, sign, or magnitude).
**Example:** The trace says the object is moving although the frame shows rest.
**Verifier signal:** Variable outside physical bounds, or mismatch with gold metadata.

### relation
**Definition:** Contact, support, containment, alignment, or spatial order is wrong.
**Example:** The trace misses that the block is on an incline.
**Verifier signal:** Contradictory relations, or relation type swapped (e.g., "on" → "above").

### force
**Definition:** A force is missing, reversed, or assigned to the wrong body.
**Example:** The normal force is not perpendicular to the surface.
**Verifier signal:** Force direction inconsistent with scenario, gravity not downward, force target not in objects.

### transition
**Definition:** The state is mostly right but the predicted change is physically invalid.
**Example:** Net force is rightward but acceleration is predicted leftward.
**Verifier signal:** Effect direction disagrees with net force, or inapplicable physical rule cited.

### intervention
**Definition:** The effect of an action or counterfactual is wrong.
**Example:** A stronger rightward push is predicted to slow the object.
**Verifier signal:** Predicted change sign disagrees with transition effect.

### temporal
**Definition:** Before/after frames are swapped or temporal order is reversed.
**Example:** The collision outcome is treated as the pre-collision frame.
**Verifier signal:** Post-transition content in state_0, or temporal markers reversed.

### unit_scale
**Definition:** Magnitude, unit, or sign convention is inconsistent.
**Example:** Centimeters are treated as meters, or velocity reported as acceleration.
**Verifier signal:** Unit mismatch in forces or answer, magnitude scaled by 10x/100x.

### faithfulness
**Definition:** The answer contradicts the model's own trace.
**Example:** The trace implies option C but the final answer is B.
**Verifier signal:** Answer value contradicts predicted change, or explicit contradiction markers.
