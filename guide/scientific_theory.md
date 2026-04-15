# Scientific Theory Compliance Guide

Based on Romero (2018), Bunge (1977), and the *Filosofía Científica*
course (UNLP, 2024).

---

## 0. What is a scientific theory?

A scientific theory is an explanation of an aspect of the natural or
social world that can be — or has been — repeatedly tested and
corroborated according to the scientific method. Its purposes are
**explanatory** and **predictive**. The strength of a theory is
measured by its capacity to produce **falsifiable predictions** about
the phenomena it claims to explain.

Formally, a theory is a set of statements closed under deduction:

    T = {s : A ⊢ s}

where A is the axiomatic base (postulates/axioms). Every statement in
T is either an axiom or a consequence of the axioms. Axioms are
primitive (non-derivable). A theory typically encompasses several
verified scientific laws, which become part of its basic hypotheses
and from which further deductions are made.

The **reference class** of the theory is the union of the reference
classes of its axioms: R(T) = ∪ R(Aᵢ). Deduction preserves reference.

### Criteria for a scientific theory

A body of knowledge qualifies as a scientific theory if it:

1. **Makes falsifiable predictions** with consistent precision across
   a broad area of inquiry.
2. **Is supported by multiple independent lines of evidence**, not a
   single foundation.
3. **Is coherent with preexisting experimental results** and at least
   as precise as prior theories.
4. **Is logically consistent** — does not contradict itself (from a
   single contradiction, any proposition follows, rendering the
   theory useless).

Additionally, scientists prefer theories that:

- Can undergo **small adaptations** to accommodate new data, thereby
  increasing predictive power over time.
- Are **parsimonious** (Occam's razor): simpler theories are
  preferred because they are more testable.
- Make **quantitative predictions**, enabling statistical methods to
  decide whether measurements fit the theory's predictions.

A theory that makes no observable predictions is not a scientific
theory. Predictions that are not specific enough to be tested are
not useful either.

### Theory vs. hypothesis vs. law

- **Hypothesis**: a conjecture corroborated by the scientific method,
  serving as a starting point (axiom) for deductions.
- **Law**: a verified regularity deducible within the theory. Laws
  become part of the theory's basic assumptions.
- **Theory**: the most rigorous, reliable, and complete form of
  scientific knowledge. It englobes hypotheses and laws into a
  coherent deductive system. Not a guess — the opposite of colloquial
  "theory."

---

## 1. Formal tools available

These are the building blocks. Use them.

### 1.1 Ordered n-tuples

    (a, b) = (c, d)  iff  a = c  and  b = d

Systems, states, meanings — all represented as n-tuples.

### 1.2 Sets, predicates, functions

- A **predicate** P : A₁ × A₂ × ... × Aₙ → S maps domains of
  objects to statements.
- R(P) = ∪ Aᵢ (reference class = union of domains).
- Properties of things are represented by **functions** on a domain M.

### 1.3 Formal language

A formal language is L = ⟨Σ, R, Ω⟩, where Σ = primitive terms,
R = rules (syntactic Sy ∪ semantic Se ∪ pragmatic Pr), Ω = universe
of discourse. If Ω = ∅, L is abstract (pure logic). If Ω ≠ ∅, L is
interpreted.

### 1.4 Deduction and theorems

Deduction is the successive application of syntactic rules. Formulas
obtained by deduction are **theorems**. A set S is **consistent** iff
¬(S ⊢ φ ∧ ¬φ).

### 1.5 Semantic concepts

- **Denotation** D : Σ → O assigns symbols to concrete objects.
- **Designation** D : Σ → C assigns symbols to concepts.
- **Reference** R : C → Ω relates concepts to objects of any kind
  (factual if Ω = O, formal if Ω = C).
- **Reference class** of concept c: [c]_R = {x ∈ Ω : R(c,x)}.
- Quantification does not change reference class.
- **Representation** assigns constructs to facts (states or changes of
  states). Rules:
  - Properties → predicates/functions.
  - Things → sets with relations/functions/operators.
  - Events (changes) → sets of singular/existential statements.
  - Laws (patterns) → sets of universal statements (equations
    constraining functions).
- **Meaning** M(c) = ⟨R(c), S(c)⟩, where R = reference, S = sense
  (logical ancestry + logical progeny within T).
- Two propositions are **synonymous** iff M(p) = M(q).

### 1.6 Vagueness

A concept is **vague** if its sense is imprecise (imprecise extension
follows). The ideal of science: produce only **exact** propositions
about the world. Eliminate vagueness through exactification.

---

## 2. Ontological framework (CESM)

### 2.1 Things and properties

- Things have **properties** (modes of being). Properties are
  intrinsic or relational.
- A thing X is modeled as X_m = ⟨M, F⟩ where F = {F₁,...,Fₙ} are
  functions on domain M, each representing a property Pᵢ.
- The universal property shared by all material objects: **energy**
  (= potential to change). Material ≡ mutable.

### 2.2 Systems

A concrete system σ is represented by:

    μ(σ) = ⟨C(σ), E(σ), S(σ), M(σ)⟩

- **C** (Composition): collection of all parts.
- **E** (Environment): items external to σ that interact with it.
- **S** (Structure): S_int (relations among components) ∪ S_ext
  (relations between components and environment).
- **M** (Mechanism): processes that make σ behave the way it does.

All four components are functions of time.

Postulate of systemic materialism: *every material object is a system
or part of a system* (Bunge 1979).

### 2.3 State space

- **State** of thing X = set of functions {Fᵢ : M → ℝ} where each
  Fᵢ represents property Pᵢ.
- **Lawful state space** S_L(X) = set of all accessible states.
- A **law statement** is a constraint on the state functions of a
  class of things (typically differential equations).

### 2.4 Events, processes, causality

- **Event** e in X = pair of states: e = (s₁, s₂) ∈ S_L(X) × S_L(X).
- **Process** p = ordered sequence of events.
- Two things **interact** iff each modifies the other's history:
  X ↔ Y iff h(X∘Y) ≠ h(X) ∪ h(Y).
- **Causality** is a relation between events, not things.

### 2.5 Emergence

P is an **emergent** property of system σ iff:

    P(σ) ∧ ∀y ∈ C(σ): ¬P(y)

Emergent properties belong to the system, not to its parts. They arise
from the constraints and interactions (structure + mechanism) among
components.

### 2.6 Information (not a thing)

Information is the propositional content of coded signals. It is a
**concept**, not a material entity. It has no energy, no independent
existence. What is destroyed is the signal, not the information.

---

## 3. Theory construction protocol

### 3.1 Phase 1 — Ontological delimitation

1. **Identify the reference class**: what concrete objects does the
   theory talk about?
2. **Apply CESM**: define C, E, S, M for your system explicitly.
   Every variable must map to a CESM component.

### 3.2 Phase 2 — State space and dynamics

1. **Define state variables**: list the measurable properties. Type
   and bound each one.
2. **Declare emergent properties**: identify system-level properties
   absent from individual components.
3. **Identify feedback loops**: where does the system observe its own
   outputs and modify its mechanisms or structure? (reflexivity /
   cybernetic regulation).

### 3.3 Phase 3 — Postulate writing (4-step protocol)

For each axiom:

1. **Empirical basis** (1 sentence): the scientific evidence or
   established knowledge the postulate rests on. Cite source.
2. **Ontological question** (italicized): *What is X?* / *What
   causes Y?*
3. **Natural-language answer** (1-2 sentences): clear, no metaphors,
   no teleology, no properties without material bearer, no
   subjective terms.
4. **Formalization** (the postulate itself): translate the answer
   into set theory, predicate logic, or equations. This **is** the
   postulate — not prose about the postulate.

#### Anti-patterns

- ✗ "We now formalize..." (advertising)
- ✗ Subjective predicates ("quality is positive")
- ✗ Properties without bearers ("information flows")
- ✗ Teleology ("the system wants to...")
- ✗ Undefined terms in formalizations
- ✗ Domain-specific jargon in ontological postulates

### 3.4 Phase 4 — Hypothetico-deductive closure

1. **Logical connection**: every postulate must share at least one
   term with another postulate. No isolated axioms.
2. **Derive theorems**: using postulates + rules of inference, obtain
   testable predictions. Format: "From Pᵢ, Pⱼ: [deduction].
   Therefore [prediction]."
3. **Bridge to experiment**: every theorem must map to a measurable
   indicator. If a theorem cannot be empirically tested, the theory
   is incomplete or metaphysical.
4. **Consistency check**: verify ¬(T ⊢ φ ∧ ¬φ). No postulate should
   contradict another.

---

## 4. Representation rules (Bunge)

When writing the theory, use these conventions:

| Real-world item    | Formal representation                              |
|--------------------|-----------------------------------------------------|
| Properties         | Predicates or functions                             |
| Things             | Sets with relations, functions, or operators        |
| Events (changes)   | Pairs of states (singular/existential statements)   |
| Laws (patterns)    | Universal statements (equations constraining Fᵢ)    |

Representation is not symmetric, not reflexive, not transitive.
Multiple representations of the same reality may be equivalent.

---

## 5. Conciseness targets

- Definition: 1 sentence + 1 formula.
- Postulate: question + evidence (1 sent.) + claim (1 sent.) +
  equation block.
- Theorem: "From Pᵢ, Pⱼ: [deduction]. Therefore [prediction]."
- No prose between postulates unless introducing a new definition.
- Target for theory section in a conference paper: ≤ 1.5 columns.

---

## 6. Checklist

- [ ] Reference class stated (what objects the theory is about).
- [ ] CESM quadruple defined, all four components explicit.
- [ ] State space defined; every variable typed and bounded.
- [ ] Emergent properties declared and attributed to the system.
- [ ] Each postulate follows the 4-step protocol.
- [ ] No postulate uses undefined terms.
- [ ] No postulate is isolated (shares terms with ≥1 other).
- [ ] ≥1 theorem derived from the postulates.
- [ ] Each theorem maps to a measurable experimental outcome.
- [ ] No meta-commentary or advertising in the theory section.
- [ ] Symbols are consistent (same symbol = same meaning throughout).
- [ ] No vague predicates — every predicate is exactified.
- [ ] Reference class of each postulate is identifiable.
- [ ] Theory is consistent: no postulate contradicts another.
