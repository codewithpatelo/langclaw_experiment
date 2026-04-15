# Writing Style Guidelines — Patricio Gerpe

These guidelines capture the author's scientific writing voice, distilled from
multiple iterations on the LangClaw paper. They are intended as a reference for
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
  of LangClaw lies not only in the mechanism itself, but in providing a testable
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

## 10. Design Goals

Every paper should aim for: **clarity**, **novelty**, **impact potential**,
**rigor**, **relevance**, and **boldness**. These are not optional; they are
the six criteria against which the writing is evaluated internally before
submission.

## 11. Philosophical and Methodological Framework

The author's work is framed within **Materialismo Complejista** — an
original theoretical framework that integrates Mario Bunge's and Gustavo E.
Romero's Systemic Materialism with Complexity Sciences and Third-Order
Cybernetics (Stafford Beer, Raul Espejo). This framing is not always made
explicit in every paper, but it informs the worldview behind the research:

- Systems are understood as emergent wholes, not reducible to their parts.
- Regulation and coordination are studied as dynamical phenomena, not only
  as engineering problems.
- Ontological questions are taken seriously: what *is* autonomy? What *is*
  coherence? What would it mean for a system to be self-regulating?

When justifying methodological decisions, connect them — where appropriate —
to ontological considerations. Example: "We chose $\Phi^*$ as a proxy not
only for its computational properties, but because it operationalizes a
question about informational integration that is relevant to the broader
debate on AI consciousness."

Use formalisms when presenting ontological premises and axioms. The style
favors making assumptions explicit rather than hiding them.

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
- Implicitly suggest **how things could be** — paint a picture of what would
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
