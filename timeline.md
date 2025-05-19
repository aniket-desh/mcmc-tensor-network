# Timeline & Goals Report: QSim 2025 Poster Preparation

## Objective
Prepare a comprehensive research poster for **QSim 2025** (deadline **July 21, 2025**) detailing theoretical foundations, implementation, and applications of a generalized tensor network contraction algorithm using MCMC methods. This report outlines the weekly milestones, development goals, and deliverables.

---

## Phase 1: Literature Deep Dive & Problem Framing
**Week of May 12–18, 2025**
- 📘 Read key papers on:
  - Glauber dynamics for partition functions.
  - Tensor network contraction.
  - Classical simulation of quantum circuits.
- 🧠 Fully understand what "random walk per iteration" means and how to implement it.
- ✍️ Begin drafting theory report.
- ✅ Deliverables:
  - Structured theory notes.
  - Annotated list of papers for citation.

---

## Phase 2: Generalization & Initial Implementation
**Week of May 19–25, 2025**
- 🔧 Formalize how to generalize contraction to arbitrary graphs.
- 💻 Implement modular data structure for arbitrary TNs.
- 📈 Start validating correctness with simple cycles and small PEPS.
- ✅ Deliverables:
  - Working TensorNetwork class.
  - Glauber sampler with burn-in and SE.

---

## Phase 3: Quantum Circuit Integration
**Week of May 26 – June 1, 2025**
- 🔬 Encode small quantum circuits as TNs (GHZ, Clifford, variational).
- 🧪 Run MCMC on these circuits.
- 🧾 Document limitations for complex or signed tensors.
- ✅ Deliverables:
  - One working quantum circuit example (e.g. 4-qubit Clifford).
  - Theory subsection on quantum application.

---

## Phase 4: Experimental Evaluation
**Week of June 2–8, 2025**
- 📊 Systematically benchmark on:
  - 2D PEPS grids.
  - MERA and random graphs.
  - Shallow quantum circuits.
- 📉 Track error vs. iterations, compare tempering.
- ✅ Deliverables:
  - Graphs/tables for error performance.
  - Result logs for poster visualizations.

---

## Phase 5: Code Optimization & Packaging
**Week of June 9–15, 2025**
- 🚀 Profile code, optimize hot paths.
- 🔄 Refactor: separate modules for sampling, evaluation, test generation.
- 🗂️ Set up reproducible experiments folder.
- ✅ Deliverables:
  - Optimized and documented code.
  - Plots comparing runtime vs. accuracy.

---

## Phase 6: Feedback & Theory Polish
**Week of June 16–22, 2025**
- 👥 Share with professor and peers for early feedback.
- 📑 Finalize theory report with full references.
- 📝 Add remarks from feedback to goals.
- ✅ Deliverables:
  - Theory report (LaTeX).
  - Revised poster outline.

---

## Phase 7: Poster Drafting
**Week of June 23–29, 2025**
- 🖼️ Create poster layout and visual hierarchy.
- 📐 Insert example networks, sampler pseudocode.
- 📍 Position figures from experiments.
- ✅ Deliverables:
  - First poster draft (PDF or Beamer).
  - Optional: share for advisor review.

---

## Phase 8: Polish and Practice
**Week of June 30 – July 6, 2025**
- 🗣️ Rehearse explanation of poster.
- 🔍 Check for typos, citation issues, formatting.
- 🧵 Print draft if possible, simulate eye-level reading.
- ✅ Deliverables:
  - Final internal version of poster.
  - Presenter notes.

---

## Phase 9: Finalization
**Week of July 7–13, 2025**
- 🖨️ Print poster or submit to printer.
- ☑️ Archive code, data, LaTeX.
- 💬 Prepare quick explanations for core questions.
- ✅ Deliverables:
  - Final poster PDF.
  - Submission to QSim portal.

---

## Phase 10: Presentation Week
**Week of July 14–20, 2025**
- ✅ Attend QSim 2025.
- 🎤 Present poster.
- 🗣️ Engage with researchers.
- 📝 Record feedback and questions for future work.

---

## Key Milestones Summary
| Date        | Milestone                                  |
|-------------|--------------------------------------------|
| May 18      | Theory draft complete                      |
| May 25      | Generalized TN code works on test cases    |
| June 1      | Quantum circuits integrated                |
| June 8      | Full benchmarks completed                  |
| June 15     | Code optimized and modular                 |
| June 22     | Final theory report and feedback integrated|
| June 29     | Poster draft complete                      |
| July 6      | Final poster polished                      |
| July 13     | Submission ready                           |
| July 21     | Poster presented at QSim                   |

---

## Research Focus Themes
- Random walk formalization.
- Tensor network generalization.
- Stochastic approximation via MCMC.
- Applications to circuit simulation and quantum verification.
- Bridging theory and prototype implementation.

Stay on track, iterate weekly, and always test theoretical claims with implementation!