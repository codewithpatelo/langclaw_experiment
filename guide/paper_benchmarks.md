# Paper Evaluation Benchmarks

Two complementary benchmarks for self-evaluating our papers before
submission. Benchmark 1 measures craft quality. Benchmark 2 measures
potential for lasting impact.

---

## Benchmark 1: Craft Quality

Rate each criterion from 1 (weak) to 5 (exceptional). A submission-ready
paper should score at least 4 on every dimension.

### 1.1 Novelty
- Does the paper introduce a genuinely new perspective, method, or
  connection?
- Is it doing something that has not been done, or is it an incremental
  improvement on existing work?
- Would a knowledgeable reviewer say "I had not thought of it that way"?

### 1.2 Clarity
- Every word is deliberate. No filler, no redundancy.
- Complex ideas are made accessible without being dumbed down.
- The structure guides the reader from problem to question to method to
  evidence to implications without friction.
- No vagueness. No ambiguity.
- All text is coherent and cohesive. Nothing seems "off".
- The paragraphs aren't isolated pieces, there is a cohesive narrative that connects each part. Each paragraph starts connecting from what happened before in previouos paragraphs. Feels like "scientific story-telling"
- No contradictions.


### 1.3 Impact Potential
- If the hypothesis holds, what changes? Which industries, institutions,
  or research agendas are affected?
- Does the paper open new directions or close a dead end?
- Is the problem socially or scientifically consequential?

### 1.4 Relevance
- Why now? What has changed (tools, data, urgency) that makes this
  question tractable or pressing today?
- Does it address a recognized gap in the literature, or does it create
  awareness of a gap that others have overlooked?

### 1.5 Boldness
- Does the paper take a risk? Does it challenge a dominant paradigm or
  connect domains that are not usually connected?
- Is the question ambitious relative to the scope of the work?
- Bold does not mean reckless: the ambition must be matched by intellectual
  humility and honest limitations.

### 1.6 Rigor
- Are claims calibrated to the evidence? No overclaiming.
- Are metrics justified (why this metric and not another)?
- Are limitations discussed seriously, not as an afterthought?
- Is the experimental design controlled for confounds?
- No vagueness, precise language.
- The logic behind claims is sound
- All math is correct


#### 1.6.1 Hyperparameter & Constant Justification (CRITICAL)

**No value in the experiment may be chosen arbitrarily.** Every numerical
constant, threshold, weight, initial condition, and hyperparameter MUST be
justified by at least one of the following:


1. **Mathematical derivation** — the value follows from the model's
   equations (e.g., setpoint ε derived from the drive function's
   fixed-point condition).
2. **Literature precedent** — cited from the original paper or a
   well-established convention in the field (e.g., discount factor γ=0.95
   is standard in TD-learning for episodic tasks; cite Sutton & Barto).
3. **Empirical calibration** — chosen via a documented procedure (grid
   search, sensitivity analysis, pilot runs) with results reported in a
   table or appendix.
4. **Principled prior** — an explicit uninformative or maximum-entropy
   choice (e.g., equal weights 1/N when no domain knowledge favours one
   component over another).
5. **Functional constraint** — the value is determined by a system
   requirement (e.g., budget hard-limit set by API cost ceiling).
6. **Relevance to the problem addressed** — the metric, parameter, or
   design decision is justified because it directly measures or controls
   a phenomenon central to the research question (e.g., CORE is chosen
   over generic perplexity because it operationalizes *conversational*
   coherence degradation, which is precisely what context collapse means
   in multi-agent discourse; or a satiation function is proportional to
   informational contribution because the theory defines epistemic need
   reduction as quality-dependent).

**What counts as a violation:**
- A constant appears in code with no comment, docstring, or paper
  reference explaining why that specific value was chosen.
- "It worked in testing" or "seemed reasonable" is NOT a justification.
- Different components use inconsistent thresholds for the same
  conceptual quantity (e.g., budget uses 0.3/0.7 while sigmoid uses θ=0.7
  without acknowledging the coupling).

**What must be documented (in code AND paper):**
- For each hyperparameter: name, value, justification method (1–5 above),
  and sensitivity (what happens if it changes ±50%).
- Reward/satiation values for each action must be outcome-dependent, not
  fixed constants — otherwise the RL component cannot learn which actions
  are genuinely useful.
- Warm-start weights for function approximators must use a documented
  initialization strategy (e.g., zero-init, Xavier/He, or domain-informed
  with explicit rationale).

**Checklist (must all be YES before submission):**
- [ ] Every constant in homeostasis.py has a justification
- [ ] Every constant in q_learner.py has a justification
- [ ] Every weight in actions.py (StimulusEvaluator) has a justification
- [ ] Every threshold in budget.py has a justification
- [ ] Every weight in delp_graph.py (Δφ*) has a justification
- [ ] LLM parameters (temperature, max_tokens) have a justification
- [ ] Satiation for all actions is outcome-dependent, not fixed
- [ ] Q-learner has an exploration mechanism (ε-greedy, softmax, or UCB)
- [ ] Sensitivity analysis is reported for critical hyperparameters

### 1.7 Reproducibility
- Can an independent researcher reproduce the experiments from the paper
  alone (code, data, seeds, hyperparameters)?
- Are results testable? Could someone design an experiment that
  *falsifies* the central claim?
- Is the methodology transparent enough that negative replications would
  be informative rather than ambiguous?
- Are dependencies, versions, environment and configuration details specified?
- Is the gap between "what the paper describes" and "what you need to
  actually run it" small?
- Do we use seeds?
- The main hyphotesis is tested through an statistical test?

---

## Benchmark 2: Breakthrough Potential

This benchmark evaluates whether the paper has the structural properties of
high-impact, field-shaping work. Not every paper needs to score high here,
but awareness of these criteria sharpens the writing.

### 2.1 Foundational Problem (Not Incremental)
A transcendent paper does not answer a known question better. It redefines
which question matters.

- Does it point to an unresolved tension (empirical or theoretical)?
- Does it expose a limit of the current paradigm?
- Does it have broad consequences if resolved?

*Example: Before AlphaFold, the question was "How do we predict structures
better?" After AlphaFold: "Why not let AI solve the complete folding
problem?"*

### 2.2 Conceptual Reframing (Ontological Shift)
This is where the real contribution lives.

- Does it introduce a new way of seeing the system?
- Does it change the units of analysis?
- Does it reorganize what counts as an explanation?

It is not just a hypothesis. It is a new language for the field.

Signs of reframing:
- Variables are redefined.
- Previous distinctions are eliminated or collapsed.
- Previously separate domains are connected.

### 2.3 Generative Mechanism (Not Just Correlation)
A breakthrough does not describe patterns: it explains how they are
generated.

- Is there a mechanistic or formal model?
- Can it simulate or predict?
- Is it transferable to other contexts?

*If you can run it (mentally or computationally), it is strong.*

### 2.4 Evidence That Closes the Case
Elegance is not enough. The evidence must be hard to dismiss.

- Are experiments designed to discriminate between competing theories?
- Are results robust, not cherry-picked?
- Ideally: counterintuitive predictions that are confirmed.

*Gold standard: "If this is true, then X (something unexpected) should
happen" --- and it does.*

### 2.5 Radical Compression
Transcendent papers tend to:

- Explain much with little.
- Unify disparate phenomena.
- Reduce apparent complexity.

*Strong signal: after reading it, it seems obvious... but nobody had seen
it before.*

### 2.6 Generality and Portability
The contribution does not stay in one case.

- Does it apply to multiple domains?
- Does it scale (conceptually or computationally)?
- Does it become cognitive infrastructure for the field?

### 2.7 Inevitable Implications
A high-impact paper does not end in itself. It opens a new research space.

- Does it generate more questions than it answers?
- Does it enable new tools or industries?
- Does it change research agendas?

*It is a branching point of knowledge.*

### 2.8 Historical Timing (Critical and Underestimated)
Many breakthroughs require:

- Data availability (e.g., large-scale datasets).
- Computational capacity.
- Prior theoretical maturity.

*Same idea, different moment --- totally different impact.*

### 2.9 Narrative Form (Yes, It Matters)
Transcendent papers are written to be:

- Inevitable (not optional).
- Anticipatory of objections.
- Guides that walk the reader toward the insight.

They are not necessarily long. They are surgical.

### 2.10 Meta-Property: Irreversibility
The definitive signal:

After that paper, the field cannot go back.

- Textbooks change.
- Questions change.
- Previous work becomes a special case.
