# Writing Style Guidelines — Patricio Gerpe

These guidelines capture the author's scientific writing voice, distilled from
multiple iterations on the Driveplexity paper. They are intended as a reference for
AI assistants, co-authors, or the author himself when drafting future papers.

---

## 1. Problem Framing First

The paper always opens from the problem, never from the solution. The reader
should understand *why this matters* before learning *what was built*.

**Sequence:**

1. **Opportunity and social implications** — Why does this problem matter beyond
   computer science? What industries, institutions, or social processes could
   benefit if this were solved? Frame it as a pressing, globally relevant issue.
2. **The problem itself** — Describe the observed phenomenon. Use hedged
   language: "tends to", "a frequently reported problem is", "agents may lose
   coherence". Do not assert universal properties as settled facts.
3. **The dominant paradigm and its limitations** — Describe how the field
   currently addresses the problem. Identify the "trendy" / "hot" / "dominant" / "established" paradigm (use hot keywords from contemporary highly cited papers) behind the standard
   approach and name its structural limitation. Use phrases like "this paradigm
   has a limitation", "this approach may address the visible symptom without
   necessarily explaining why..."
4. **An apparently unrelated phenomenon** — Introduce a concept from a
   different discipline (biology, economics, physics, etc.) that *at first*
   seems disconnected from the problem. Present it as what it is: an
   observation from another domain, not yet connected to the paper's concern.
   "A well-established mechanism in [field] is... In that domain, regulation
   does not depend on external control; it emerges from..."
   **Crucially, do not yet frame it as a solution.** It is still just a
   phenomenon. The transformation happens in the next step.
5. **The scientific question transforms the phenomenon into an approach** —
   This is the narrative pivot. By asking the right question ("what if X could
   be applied to Y?"), the apparently unrelated phenomenon becomes an
   unexploited approach. The question is what bridges the two domains.
   Example: homeostasis is just biology — until you ask "what if multi-agent
   coherence could be regulated endogenously rather than imposed externally?"
   At that point, homeostasis becomes a candidate operational principle.
   This two-step move (unrelated phenomenon → right question → unexploited
   approach) is the signature of the author's lateral thinking style.
6. **Only then, the proposal** — Introduce the framework/method as a vehicle to
   *explore* the question, not as a definitive answer.

## 2. Tone and Claims

- **Every word is deliberate.** The prose is concise, clear, and forceful,
  yet capable of making complex ideas accessible. No filler, no redundancy, nothing trivial.
  If a sentence can be shorter without losing meaning, it should be.
- **Observe, contemplate, frame, ask, then propose.** Never jump to claims.
- **Avoid hard claims.** Replace "X suffers from Y" with "X tends to exhibit Y"
  or "a recurrent problem reported in X is Y".
- **Hedged language is not weakness.** It is precision. Use: "could be explained
  by", "this suggests", "is consistent with", "tends to", "may", "we are not
  aware of previous studies that..."
- **Never say "No prior work has done this."** Say "To the best of our
  knowledge, this perspective has not been evaluated as..."
- **Never say "We prove" or "We demonstrate."** Say "We observe", "We evaluate",
  "The results suggest", "The data is consistent with".

## 3. Paradigm Clashes

A hallmark of this writing style is explicitly naming the tension between
paradigms. Not just "method A vs method B", but "two ways of understanding
orchestration" or "the dominant view assumes X, yet the phenomenon to be
explained is Y".

This framing elevates a technical comparison to a conceptual contribution.

## 4. Scientific Question and Implications

- The research question must be stated explicitly, clearly, and early.
- Always explain *why the question matters*, not just what it asks.
- Connect the question to broader implications: "If X depends on Y rather than
  Z, then many current strategies may be addressing the symptom rather than the
  cause."

## 5. Antecedents and State of the Art

- Always show knowledge of the pioneers in the field.
- Describe the evolution: foundational work, current state of the art, and
  emerging directions.
- Use temporal markers: "More recently, work on... has begun to explore..."
- Extract unresolved challenges from the literature: "This leaves open the
  question of..."

## 6. Experimental Writing

### Before the experiment

- **Justify every metric.** If you use a metric, explain why that one and not
  another. Example: "We chose tau-bench because it captures logical
  connectivity rather than surface-level fluency. Alternative metrics such as
  ... would measure ... but would not capture..."
- **State expectations explicitly.** Before presenting results, write what you
  expect to observe and why: "We expect three patterns. First, if the hypothesis
  is correct, X should... Second, we expect Y to... Third, with symmetric
  roles..."

### After the experiment

- **Report what was found, then interpret.** Do not mix observation with
  explanation in the same sentence.
- **Handle surprises honestly.** When results contradict expectations, say so
  explicitly: "Contrarily to this expectation, ...", "This result is, at first
  glance, contrary to..."
- **Explain the surprise.** Immediately follow with a candidate explanation:
  "This could be explained by...", "This difference may be attributable to..."
- **Ask the data questions.** Frame analysis as inquiry: "This design allows us
  to ask the data a constrained question: under identical conditions, which
  regime better sustains..."

## 7. Contributions

- List contributions explicitly near the end of the Introduction.
- Frame them as what the paper *does*, not what it *proves*.
- Connect each contribution back to the research question.

## 8. Conclusion

- Restate the research question.
- Summarize what was found, with appropriate hedging.
- Discuss broader implications (what would change if the hypothesis holds at
  scale).
- End with the value of *asking the question*, not just answering it: "the value
  of Driveplexity lies not only in the mechanism itself, but in providing a testable
  way to ask whether..."

## 9. Characteristic Phrases

These are phrases and patterns the author naturally gravitates toward:

- "El avance de [paradigm/tool/discipline] ... Esto plantea [dilemmas,
  limitations, risks, underexploited opportunities, unresolved challenges]..."
- "More recently, studies have begun to emerge that specifically address..."
- "A well-established mechanism in [field] that remains unexploited in this
  context is..."
- "This motivates a broader question: what if...?"
- "We expect that... / It is expected that..."
- "Contrarily to what was expected..."
- "This could be explained by..."
- "This suggests that... rather than..."
- "The question is not only [technical], but also [conceptual/paradigmatic]."
- "If [condition holds], then [broader implication]."

## 10. Recurring Conceptual Vocabulary

These are concepts the author returns to across papers. They are not
keywords to sprinkle decoratively — they are the conceptual primitives
through which problems are analyzed. When writing, check whether the
argument can be sharpened by grounding it in one of these:

- **Sistema**: the fundamental unit of analysis. Everything concrete is
  a system or a component of one (Bunge). Analysis always asks: what is
  the composition, environment, structure, and mechanism?
- **Recursión**: viable structures contain and are contained in viable
  structures (Beer). Patterns repeat at different scales. If a
  regulatory mechanism works at one level, ask whether it applies
  recursively.
- **Nivel de recursividad**: the specific scale at which a recursive
  pattern is being observed. The same VSM structure applies at the team,
  department, organization, and ecosystem levels. Always specify which
  recursion level the analysis operates on.
- **Variedad** (variety): the cybernetic measure of complexity (Ashby).
  A regulator must match the variety of what it regulates. Insufficient
  variety means loss of control; excess variety means waste.
- **Perturbación** (disturbance): external or internal events that push
  a system away from equilibrium. The interesting question is not
  whether disturbances occur, but how the system absorbs or amplifies
  them.
- **Sensor**: the interface between a system and its environment or
  internal state. What a system can regulate depends on what it can
  sense. Absent sensors, regulation is blind.
- **Emergencia**: higher-level properties that arise from lower-level
  interactions but are not reducible to them. Emergence is lawful, not
  mystical (Bunge). The question is always: what mechanism produces the
  emergent property?
- **Convergencia**: the tendency of initially disparate processes,
  disciplines, or technologies to meet. Often the site of innovation.
- **Superveniencia**: a property supervenes on another if there can be
  no change in the former without a change in the latter. Useful for
  clarifying ontological dependencies without claiming reduction.
- **Materia**: the commitment to physicalism. All concrete entities are
  material. Information, computation, and regulation are properties of
  material systems, not free-floating abstractions.
- **Dialéctica**: the productive tension between opposing forces or
  paradigms. Not Hegelian idealism — materialist dialectics in Bunge's
  sense: contradiction as a driver of structural change in concrete
  systems.
- **Sinergia**: the interaction of components producing an effect
  greater than the sum of individual contributions. The quantitative
  signature of emergence.
- **Tendencia**: the directionality of a process. Systems exhibit
  tendencies (toward equilibrium, toward collapse, toward
  self-organization) that can be observed, measured, and sometimes
  redirected.
- **Feedback**: information about the output of a process that is fed
  back to regulate future behavior. Negative feedback stabilizes
  (homeostasis); positive feedback amplifies (runaway growth, collapse).
  Always ask: is the loop closed? What signal closes it?
- **Feedforward**: anticipatory regulation based on predicted
  disturbances rather than observed outputs. A system with feedforward
  acts before the error occurs, not after. More fragile than feedback
  (depends on model accuracy) but faster.
- **Corrección**: the regulatory act of reducing the gap between actual
  and desired state. Correction presupposes a sensor (to detect the
  gap), a comparator (to evaluate it), and an effector (to act on it).
  In Driveplexity, satiation is the correction mechanism: it reduces the
  deficit proportionally to argument quality.
- **Viabilidad**: the capacity of a system to maintain its identity and
  adapt to a changing environment over time (Beer). A system is viable
  if it can survive — not just function momentarily. Viability requires
  balancing internal stability with external adaptation.
- **VSM (Viable System Model)**: Beer's recursive model specifying the
  five subsystems necessary and sufficient for viability. System 1:
  operations (primary activities). System 2: coordination (anti-
  oscillatory dampening). System 3: control (internal regulation,
  resource allocation). System 4: intelligence (environment scanning,
  adaptation, future planning). System 5: policy (identity, values,
  ultimate authority). The model is a lens for analyzing any autonomous
  system — from a cell to a corporation to a multi-agent simulation.
  When designing or analyzing a system, ask: which of the five
  functions is missing, underdeveloped, or disconnected?
- **Autoorganización**: the capacity of a system to generate internal
  order without external instruction. The central question of this
  research program: under what conditions does coordination arise
  endogenously?

## 11. Design Goals

Every paper should aim for: **clarity**, **novelty**, **impact potential**,
**rigor**, **relevance**, and **boldness**. These are not optional; they are
the six criteria against which the writing is evaluated internally before
submission.

## 11. Philosophical and Methodological Framework

The author's work is framed within **Materialismo Complejista** — an
original theoretical framework. Its components and intellectual lineage
are as follows:

### Mario Bunge (1919–2020)
From Bunge we take three pillars:
- **Scientificism**: the commitment to the scientific method as the most
  reliable path to knowledge, rejecting pseudoscience and obscurantism.
- **Emergentism**: the thesis that higher-level properties (life, mind,
  society) are real and emergent — they arise from lower-level processes
  but are not reducible to them. A system is more than the sum of its
  parts, and the "more" is not mystical but lawful.
- **Systemism**: the ontological position that everything concrete is
  either a system or a component of a system. Systemism rejects both
  individualism (only parts matter) and holism (only the whole matters),
  proposing instead that explanation requires understanding the
  composition, environment, structure, and mechanism of systems.

Key works: *Treatise on Basic Philosophy* (8 vols.), *Emergence and
Convergence*, *Chasing Reality*.

### Gustavo E. Romero
From Romero we take the **cátedra de filosofía científica** — a rigorous,
physicalist approach to foundational questions. Romero extends Bunge's
systemism into contemporary physics and philosophy of science, insisting
on ontological clarity: what *exists*, what *is* a property, what *is*
a law. His work models how to take ontological questions seriously
within a scientific (not speculative) framework.

Key works: *Scientific Philosophy*, *Conceptual Issues in Astrophysics*
(with Gustavo Romero & Gabriela Vila).

### Stafford Beer (1926–2002)
From Beer we take the **Viable System Model (VSM)** — a recursive
cybernetic model of organizational viability. The VSM identifies five
interacting subsystems (Systems 1–5) that any autonomous system must
exhibit to survive in a changing environment. Key principles include:
- **Requisite variety** (Ashby's Law): a regulator must have at least as
  much variety as the system it regulates.
- **Recursive structure**: every viable system contains and is contained
  in a viable system.
- **Autonomy with cohesion**: viable systems balance local autonomy
  (System 1) with global coherence (System 5).

Beer belongs to **management cybernetics** (second-order cybernetics
applied to organizations). His work is foundational, not third-order.

Key works: *Brain of the Firm*, *The Heart of Enterprise*, *Diagnosing
the System for Organizations*.

### Raul Espejo
From Espejo we take:
- The **Viplan Method**: a practical methodology for diagnosing and
  designing organizations using the VSM, including tools for unfolding
  complexity and distributing discretion.
- **Organizational transformation as a cybernetic process**: Espejo
  extended Beer's VSM into a methodology for real-world institutional
  change, emphasizing stakeholder participation, self-construction of
  desirable social systems, and the governance of complexity.
- **The Cybersyn experience**: Espejo was operational director of Chile's
  Project Cybersyn (1971–1973) under Beer's scientific direction — a
  pioneering attempt to apply real-time cybernetic management to a
  national economy.

Key works: *Organizational Transformation and Learning* (with Schuhmann
et al.), *Organizational Systems* (with Reyes), *The Viable System
Model: Interpretations and Applications* (ed. with Harnden).

### Complexity Sciences
From the broader complexity sciences tradition we integrate:
- Nonlinear dynamics, emergence, self-organization.
- Complex adaptive systems (Holland, Kauffman).
- Dissipative structures and far-from-equilibrium thermodynamics
  (Prigogine).

### Integration: Materialismo Complejista
The synthesis is: **Bunge's systemism and emergentism** provide the
ontological foundation (what exists and how it is organized). **Romero's
scientific philosophy** provides the epistemological discipline (how to
reason rigorously about ontological questions). **Beer's VSM** provides
the regulatory architecture (how viable systems maintain identity
through internal regulation). **Espejo's methodology** provides the
operational bridge between theory and practice. **Complexity sciences**
provide the dynamical vocabulary (attractors, phase transitions,
self-organization).

This framework is not always made explicit in every paper, but it
informs the worldview behind the research:

- Systems are understood as emergent wholes, not reducible to their parts.
- Regulation and coordination are studied as dynamical phenomena, not only
  as engineering problems.
- Ontological questions are taken seriously: what *is* autonomy? What *is*
  coherence? What would it mean for a system to be self-regulating?
- Viability is understood as a property that requires both internal
  regulation and external adaptation — not just one or the other.

When justifying methodological decisions, connect them — where appropriate —
to ontological considerations. Example: "We chose $\Phi^*$ as a proxy not
only for its computational properties, but because it operationalizes a
question about informational integration that is relevant to the broader
debate on AI consciousness."

Use formalisms for the presenting ontological premises and axioms of the experiment. The style favors making assumptions explicit but logically sound, derived from scientific facts rather than hiding them. Abductive reasoning is key for this.

## 12. Thinking Modes

The writing uses three cognitive modes explicitly:

- **Lateral thinking**: connecting apparently unrelated domains to generate
  new approaches (biology → multi-agent orchestration).
- **Systems thinking**: analyzing how parts interact to produce emergent
  behavior, not just describing components in isolation.
- **Analogy thinking**: drawing structural parallels between domains to
  transfer explanatory power.

## 13. Epistemic Attitude

- **Bold but intellectually humble.** We pursue breakthrough ideas, but we
  assume we are wrong about something — we just don't know what yet.
- **Methodological skepticism.** Every method, metric, and design choice is
  treated as provisional and open to revision.
- **Critical thinking with a search for serendipity.** We are rigorous but
  also attentive to unexpected patterns in the data — epiphanies that
  challenge our initial framing.
- **Claims are never definitive.** They are informed by evidence, not by
  conviction.

## 14. Purpose and Timeliness

Every paper must answer two questions explicitly:

- **Para qué** (what for): Why does this research matter? Who benefits? How
  could it transfer to industry or social practice? The work should be
  pragmatic — not just theoretically interesting but potentially useful.
- **Por qué ahora** (why now): What has changed recently — in tools,
  data availability, computational capacity, or societal urgency — that
  makes this question tractable or pressing *today* in a way it was not
  before?

## 15. Literature and Antecedents

Beyond what section 5 already covers:

- Identify who *coined the term* or *introduced the concept*. Give credit
  to the intellectual origin, not just the most recent citation.
- Seek **underexploited approaches**: ideas that were proposed but not
  widely adopted, that could be revindicated with today's tools.
- Seek **forgotten approaches**: older work that fell out of fashion but
  whose core insight remains valid and unexploited.
- Highlight **new tools** that make previously intractable approaches
  feasible now (e.g., LLMs making it possible to test argumentation
  theories at scale).

## 16. Evidence-Based Framing

When framing why a problem matters, use evidence from authoritative sources:
"According to the United Nations...", "The World Economic Forum estimates
that...", "Recent studies suggest that X affects Y million people." This
grounds the motivation in data, not in assertion.

## 17. Social Orientation and Externalities

- The work has a social dimension: AI for good, responsible AI, algorithmic
  fairness, green computing, safety, and accessibility are not afterthoughts.
- Explicitly consider both **positive and negative externalities** of the
  proposed approach. What could go wrong? What unintended consequences could
  arise? What ethical considerations apply?
- Visionary. Implicitly suggest **how things could be** — paint a picture of what would
  change if the approach works at scale, without overclaiming.

## 18. Persuasion Strategy

The persuasion is purely scientific, based almost exclusively on:

- **Logos**: logical argumentation, formal reasoning, evidence.
- **Kairos**: timeliness — why this question matters *now*.
- **Ethos**: demonstrated knowledge of the field, honest treatment of
  limitations, intellectual humility.

A minimal and subtle use of **pathos** is acceptable to connect emotionally
with the problem ("the reliability of AI systems in socially consequential
settings"), but it must always remain rigorous and objective. Never
sentimental, never alarmist. 

## 19. What to Avoid

- Marketing language, hype, or overclaiming.
- Listing features without explaining why they matter.
- Presenting a solution before the reader understands the problem.
- Using "novel", "groundbreaking", "first-ever" without strong evidence.
- Treating limitations as afterthoughts; they should be discussed seriously.
- Generic paper-speak: "In this paper we propose..." as the first sentence.
  Always open with the problem or the phenomenon.
- Ignoring externalities (positive or negative) of the proposed approach.
- Avoiding ontological questions when they are relevant to the design.
- Treating the research as purely theoretical when it could transfer to
  practice (pragmatism matters).
- Definitive claims where provisional ones would be more honest.
- Pathos-heavy rhetoric; persuasion must rest on evidence and logic.
