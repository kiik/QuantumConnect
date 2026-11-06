# Designing Robots for the Human Brain: UX in Autonomous and Non-Autonomous Systems

**Speaker:** Sara Esperto
**Notes added:** 2026-11-06
**Topic:** Shared mental models in human–robot interaction (HRI)

---

## 1. The core problem

- Robots used to be judged on **capability**: can it navigate, grasp, inspect?
- Now that they work next to people, a second question matters as much: **can the people around it
  predict what it will do?**
- Many automation failures aren't hardware or algorithm failures. The operator expected one thing
  and the machine did another.
- Human-factors name for this: **automation surprise**. Classic examples come from aviation, where
  pilots asked:
  - "Why did it do that?"
  - "What's it doing now?"

## 2. What a shared mental model is

- **Mental model**: your internal simulation of how something works, used to predict and explain it.
  From cognitive psychology (Johnson-Laird, Norman).
- **Shared mental model (SMM)**: from team research (Cannon-Bowers, Salas and others, 1990s). Teams
  perform well when members hold overlapping models of:

| Model of the... | Covers |
|---|---|
| **Task** | goals, procedures, what "done" looks like |
| **Team** | who does what, who knows what, each member's capabilities and limits |
| **Situation** | what's happening right now and what comes next |

- Good human teams coordinate **implicitly**. A surgeon holds out a hand and the nurse knows which
  instrument to pass.
- SMM research in HRI asks how to get that kind of implicit coordination between a person and a
  robot.

## 3. Why it's harder with robots

1. **Asymmetric legibility.** People read each other's gaze, posture and hesitation. A robot's
   internal state (planner output, sensor confidence, battery level) is invisible unless someone
   designs a way to show it.
2. **Anthropomorphic misreading.** People project human or animal intentions onto machines,
   especially animal-like ones such as a four-legged "robodog". Those projections set expectations
   the machine may not meet.
3. **The model has to run both ways.** The human needs a model of the robot. Ideally the robot also
   has a model of the human: their intent, workload and likely next action.
4. **Trust calibration.** The goal is *appropriate* trust, not maximum trust (Lee & See, 2004).
   - Over-trust → misuse, complacency.
   - Under-trust → disuse: people override or ignore a useful system.
   - A well-aligned SMM is what lets trust track the robot's real reliability.

## 4. Design levers: where UX and cognitive psychology come in

The practical levers that turn SMM theory into a working system:

- **Legibility and predictability.** Motion and behaviour signal intent *before* the action happens
  (Dragan, legible motion). E.g. slow down and turn the "head" toward a target before approaching it.
- **Transparency at the right level.** Endsley's situation-awareness model has three levels:
  1. what the robot **perceives**,
  2. what it **understands**,
  3. what it **will do next**.

  Interfaces should support all three without overloading the operator.
- **Consistency and conventions.** Same signals for the same states every time (lights, sounds,
  screen colours). Predictability is a UX property as much as an engineering one.
- **Feedback and feedforward.**
  - Feedback: confirm what the robot received ("command acknowledged").
  - Feedforward: preview what it will do next ("next: inspecting sector 3").
- **Graceful failure and handover.** Make uncertainty and limits visible ("low confidence", "can't
  reach this area") so the human knows when to step in.
- **Training and onboarding.** Mental models are partly built *before* use, through training,
  simulation and documentation.

## 5. The CERN context

CERN is a demanding test case:

- radiation, a confined 27 km tunnel, high-value equipment, limited human access;
- operators often supervise robots **remotely through screens, with no line of sight**, so the
  whole SMM burden falls on the interface.

### TIM (Train Inspection Monorail)

- Small autonomous train on a rail along the LHC tunnel ceiling; in use since about 2016.
- Carries cameras and sensors (radiation, oxygen, temperature) so people don't have to enter the
  tunnel.
- SMM questions: does the operator know **where TIM is**, **what it has checked**, and **why it
  stopped**?

### CERN's Robodog

- Legged robot adapted for inspection and intervention in hard-to-reach or radioactive areas.
- Legs bring mobility, but also **less predictable motion** and **strong anthropomorphic
  expectations**.
- So legible behaviour and clear status communication matter especially, both for remote operators
  and for staff who meet it on site.

## 6. Takeaway

> A robot that is capable but unpredictable is unsafe in practice.

Safe HRI comes from designing for **mutual predictability**: the human can anticipate the robot,
and ideally the robot can anticipate the human. That combines:

- team cognition research (shared mental models),
- cognitive psychology (situation awareness, trust calibration),
- UX practice (legibility, transparency, feedback).

## Key names to look up

| Name | Known for |
|---|---|
| Cannon-Bowers & Salas | shared mental models |
| Endsley | situation awareness (3 levels) |
| Lee & See (2004) | trust in automation, calibrated trust |
| Norman | mental models and design |
| Dragan | legible robot motion |

## Sources

- *Meet TIM, the LHC tunnel's robot* (CERN)
- *Train Inspection Monorail* (CERN Knowledge Transfer)
- *Introducing CERN's Robodog* (CERN)
- *CERN is training robot dogs* (The Register)
- *The robots of CERN* (Symmetry)
- *On the effect of AR multi-user interfaces and shared mental models on human-robot trust*
  (Imperial)
- *Trust and Acceptance of Multi-Robot Systems "in the Wild"* (arXiv)
