# Driveplexity — Theoretical Grounding

This document is the public-facing version of the theoretical
framing that underpins Driveplexity. For the author's canonical,
authoritative statement (used as the internal reference whenever
the paper's theory section is edited) see
[`../guide/TAH_axiomas.md`](../guide/TAH_axiomas.md).

---

## 1. Axioms of Homeostatic Autonomy (AAH)

Driveplexity is grounded on three **ontological axioms** that state
the minimal commitments the design makes about what it means for an
agent to be autonomous, what makes it act, and what makes an action
valuable. The axioms are general: they describe *any* homeostatic
agent, not only the specific MAS-LLM implementation used in this
paper.

### A1 · Autonomy (Ashby, 1956; Beer, 1972)

*What is autonomy?* The policy of the agent depends on its internal
state.

$$
\mathrm{Aut}(a_i) \iff \exists\,\delta_i \in S_L(a_i):\ \pi_i = \pi_i(\delta_i(t)).
$$

One-line form: *sustain quality action over time under internal
regulation*.

### A2 · Drive (Sterling & Eyer, 1988)

*What makes an agent act?* An internal drive that grows with
deviation from equilibrium and whose magnitude monotonically
determines activation probability.

$$
D(\varepsilon) = 0,\ D'(\delta_i) > 0\ \forall\,\delta_i > \varepsilon,\ p_i(t) = \sigma(D(\delta_i(t))).
$$

One-line form: *growing pressure to satisfy a deficit in the face
of perturbations*.

### A3 · Quality gate (Ryan & Deci, 2001; Huta & Waterman, 2014)

*What makes an action valuable?* An effective change in the state
of the regulated environment. Let `g(e(t), e(t+1)) ≥ 0` be the
measure of that change; the drive decreases proportionally.

$$
\Delta \delta_i = -\alpha \cdot g(e(t), e(t+1)),\quad \alpha > 0.
$$

One-line form: *a quality action is one that satisfies the
deficit*.

---

## 2. Derived prediction (Pr1)

> Given an agent exposed to perturbations whose drive grows with
> deviation from equilibrium and decreases only after effective
> change, one would expect sustained quality action to emerge,
> with activation probability increasing in the deficit.

Under bounded reward `r` and a learning rate `η_t` satisfying the
Robbins–Monro conditions (`Σ η_t = ∞`, `Σ η_t² < ∞`), the TD(0)
update on `Q(·)` converges to a fixed point of the Bellman operator
(Keramati & Gutkin, 2014). The Driveplexity implementation uses
off-policy linear function approximation, which places the TD loop
in a **deadly triad** regime: the axioms do not *guarantee*
convergence there, so the paper scopes the Q-learner as an
experimental extension whose convergence is an empirical open
question, not a theoretical assumption.

---

## 3. Relation to HRRL

The paper uses **HRRL (Homeostatic Regulation via Reinforcement
Learning)** — Keramati & Gutkin (2014) — only as a *theoretical
antecedent* for the drive formulation and its reward-as-drive-
reduction interpretation. The experimental system reported in the
paper is **Driveplexity**, not a port of the vanilla HRRL setting to
a language-model context: Driveplexity adds AAH-driven commitments,
a deliberation-specific quality gate over an Abstract Argumentation
Framework, an explicit coupling with the Viable System Model for
social organisation, and an experimental TD(0) extension whose
convergence is reported rather than assumed.

Reviewers should not read a claim about Driveplexity as a claim
about HRRL, and vice versa.

---

## 4. Role of the Viable System Model (VSM)

Each faction is mapped to the five subsystems of the VSM (S1
operation, S2 coordination, S3 control, S4 intelligence, S5
policy). VSM is adopted as the **minimal complete unit of social
organisation**: five roles whose interaction could plausibly give
rise to emergent social dynamics beyond individual regulation.

The paper does not claim to characterise those emergent patterns in
the submitted short version; it only sets up the architecture that
would allow a follow-up study to look for them.

---

## 5. What this framing commits the design to

From A1–A3 the following design decisions follow directly:

1. A per-agent internal scalar (`δ_i`) that is *written* by the
   environment and *read* by the policy.
2. A monotone map from `δ_i` to activation probability — any
   specific form satisfying `D(ε) = 0`, `D'(δ) > 0` works; the
   implementation uses `σ(k(δ − θ))`.
3. A reward that is a *decrease in drive* produced by an action,
   not an external signal.
4. A quality measure `g` over the regulated environment — in this
   paper, a network-theoretic proxy over the argumentation graph.
5. A clean separation between *triage* (who acts, decided by the
   sigmoid gate) and *action choice* (what they do, decided by the
   Q-learner or any other policy).

The two control conditions discriminate the minimal pieces of this
commitment:

- **DPLXY-no-Q** removes the Q-learner. If the macro pattern
  survives, the effect cannot be attributed to learning and is
  consistent with A1–A3 alone.
- **LangGraph-informed** removes the endogenous triage but keeps
  the information it would use (the same five features and `δ_i`).
  If the gap with DPLXY remains, the effect cannot be reduced to
  information access and is consistent with the **regulatory
  closure** (A2+A3) being the operative mechanism.

Both controls are preliminary (`n = 1` and `n = 2` respectively)
and are reported in the paper as directional evidence, not proof.
