# Intelligent Data Systems and AI for Autonomous Space Technology

**Speaker:** Sudeshna Sengupta (Software Engineer at CERN)
**Event:** Fall Campus 2026, two parts
**Notes added:** 2026-11-06
**Slides:**
[Part 1: Satellite Data and the Downlink Bottleneck](https://indico.cern.ch/event/1645790/contributions/7037603/attachments/3374149/6056966/Part%201%20Satellite%20Data%20and%20the%20Downlink%20Bottleneck.pptx.pdf) ·
[Part 2: Machine Learning on the On-Board Computer](https://indico.cern.ch/event/1645790/contributions/7037602/attachments/3375121/6059061/Part%202%20Machine%20Learning%20on%20the%20On-Board%20Computer.pptx.pdf)
**Demo kit:** <https://github.com/Sudeshna2499/Fall-Campus-2026-Cloud-Detection-Demo>

> **One-line summary.** A satellite measures far more than it can send home, so software on board
> has to decide what is worth sending, and we need ways to trust those decisions.

Lines marked **Background** are added explanations, not slide content.

---

# Part 1: Satellite data and the downlink bottleneck

## 1.1 The opening question

> If you could send only a handful of events on each pass, which ones would you send?

A satellite measures far more than it can send home. The whole lecture follows from that.

## 1.2 Why satellite data matters

| Area | Why it matters |
|---|---|
| **Weather and climate** | 60% of Essential Climate Variables (ECVs) can be measured from space (ESA Climate Change Initiative). |
| **Disaster response** | Rapid access to satellite data for floods, earthquakes and volcanic eruptions. |
| **Land and sea** | Farming decisions (e.g. when to harvest before rain) and maritime safety. |
| **Space itself** | Space weather and debris threaten the satellites we depend on, so we have to measure them. |

### When the data is about space itself: the Starlink example

- Solar eruptions disturb the upper atmosphere → **drag rises** → low orbits **decay faster**.
- **February 2022:** 38 of 49 Starlink satellites were lost after launch. A *minor* geomagnetic
  storm raised drag and reduced stability (source: CIRES, University of Colorado Boulder).
- Better warning needs measurements made **in orbit**: particles, plasma waves, dust impacts.

## 1.3 How satellite data is collected

The data chain, from instrument to ground:

```
Sense ──► Digitise ──► Process and store ──► Packetise ──► Downlink
```

| Step | What happens |
|---|---|
| **Sense** | An instrument measures images, particles, radio waves or fields. |
| **Digitise** | An analogue-to-digital converter turns the signal into samples. |
| **Process and store** | The on-board computer buffers, processes and stores. |
| **Packetise** | Data are packed into telemetry packets. |
| **Downlink** | The radio sends them while a ground station is in view. |

Two kinds of data travel down:

- **Payload data**: what the mission exists to measure.
- **Housekeeping data**: the satellite's own health (temperatures, voltages, attitude).

### Example: on-board signal processing for a plasma wave receiver

The stages shown on the slide:

```
Plasma wave receiver → Raw data input (digitised) → Digital filtering → Spectral analysis
                     → Feature extraction → Data compression → Ground station
```

(The order is read from the slide's diagram layout.) The idea is that reduction happens step by step
on board, so what reaches the ground is features and compressed data, not raw samples.

## 1.4 Small platforms, big appetite

**CubeSats** are built from 10 cm units ("U"). Φsat-1 and Φsat-2 are **6U**; OPS-SAT was about
30 cm tall.

Satellite subsystems:

| Subsystem | Role |
|---|---|
| Power | solar panels and battery |
| Attitude control | pointing and stability |
| Communications | radio link to the ground |
| Payload | the science instrument |
| Structure and thermal | frame and heat control |
| **On-board computer (OBC)** | **the main focus of the lecture** |

**What the OBC does:**

- takes commands from the ground;
- schedules and monitors subsystems;
- stores and packages data;
- can run payload software (this is where on-board AI lives).

## 1.5 The downlink is the bottleneck

| Figure | Meaning |
|---|---|
| **2.7 TB** | generated per day by one Sentinel-2 satellite |
| **~1 TB** | per day received by the ground stations, about **37%** |
| **5–50 kbps** | a typical "Long Range" link on a LEO satellite; S-band reaches up to **2 Mbps** |

- Data can wait **up to 30 days** in the on-board buffer.
- Contact is limited to **passes** over a ground station.

Source: OrbitChain, arXiv:2508.13374 (preprint, not peer reviewed).

> **Background: a sense of scale** (my arithmetic, not from the slides).
> - 1 TB/day is about 8 × 10¹² bits / 86 400 s ≈ **93 Mbps averaged over the whole day**. So
>   Sentinel-2's downlink is much faster than the small-satellite links on the same slide.
> - At 2 Mbps (S-band, best case on the slide), sending 2.7 TB would take ≈ 1.1 × 10⁷ s ≈
>   **125 days** of *continuous* contact. The real time would be much longer, because contact only
>   happens during passes.
> - **LEO** (low Earth orbit) satellites circle the Earth roughly every 90–100 minutes, so a given
>   ground station sees each one only for short windows of a few minutes.

## 1.6 Send everything, or decide on board?

| | Send everything | Decide on board |
|---|---|---|
| **Idea** | simple and flexible: decide later on the ground | screen, prioritise or compress before sending |
| **Upside** | no information thrown away by the satellite | faster alerts and fewer wasted bytes |
| **Downside** | buffers fill, old data is delayed or lost; most of the link carries uneventful data | **risk: a wrong decision discards data for good** |

The four things on-board software can do (these return in Part 2):

| Job | Example |
|---|---|
| **Screen** | discard unusable data, e.g. cloudy images |
| **Detect** | find ships, fires, events |
| **Prioritise** | rank what goes first |
| **Compress** | shrink what is sent |

## 1.7 Missions that put AI on board

| | **OPS-SAT** | **Φsat-1** | **Φsat-2** |
|---|---|---|---|
| **When** | 2019–2024; mission ended May 2024 | launched Sep 2020 | launched Aug 2024 |
| **Platform** | ESA experimental CubeSat, ~30 cm tall | 6U CubeSat with a hyperspectral camera | 6U, built by Open Cosmos |
| **Compute** | Cyclone V SoC: dual-core ARM Cortex-A9 + FPGA, running Linux | Intel Movidius Myriad 2 processor | (not given on slide) |
| **AI use** | experimenters uploaded their own software | detects clouds on board and discards heavily cloudy images | apps at launch: cloud detection, street maps, vessel detection, image compression. Uploaded later: marine anomaly and wildfire detection |
| **Why** | an in-orbit test bed | **over 30% of Sentinel-2 images are cloudy** | |

## 1.8 Live demo: an Arduino as the on-board computer

A toy version of Φsat: the same cloud-detection idea, on a hobby board, with made-up pixels.
**All data are synthetic.**

**Setup:**

- Synthetic image tiles of **8 × 8 pixels** in red, green, blue and near-infrared (NIR).
- **Five features per tile:** brightness, texture, whiteness, vegetation index (NDVI), NIR level.
- Two classifiers on the same five features:
  1. a simple **brightness rule** (threshold);
  2. a **tiny neural network**: **75 parameters, 300 bytes**.
- Policy: **cloudy tiles are discarded; unsure tiles are sent.**

> **Background: NDVI.** Normalised Difference Vegetation Index =
> (NIR − Red) / (NIR + Red). Healthy vegetation reflects strongly in near-infrared and absorbs red,
> so it scores high. Clouds are bright in *all* bands, so they score near zero. That is why
> spectral features like this help tell clouds from bright ground.

**Reading the demo window** (simulated pass with synthetic signals):

| Panel | What it shows |
|---|---|
| **Scene and maps** | Green = clear and sent · Grey = cloud, discarded · **Red = clear tile lost** · Orange = cloud sent |
| **Features** | Bright sand sits near the rule's line; *whiteness* helps tell it from cloud. |
| **Bytes and scores** | Data to downlink versus sending every tile, and clear tiles lost. |

Red is the costly error (good data destroyed). Orange is the cheap error (some bytes wasted).

### Results on the simulated scenes

| Figure | Meaning |
|---|---|
| **36%** | fewer bytes to downlink, because about 36% of tiles are cloud |
| **1.6%** | of clear tiles wrongly discarded by the **brightness rule** |
| **0.7%** | of clear tiles wrongly discarded by the **network** |

**Read with care:** the network's gain is modest, and cloud recall is about **98–99%** (the slide
gives this as a range over simulated passes). The pixels are synthetic and the speaker built
bright sand into the simulator, so the gap between rule and network reflects the simulator, not real
imagery.

## Part 1 recap

1. Satellite data underpins weather, climate, disaster response and space safety.
2. Data are sensed, digitised, processed on board and sent during passes.
3. The downlink is the bottleneck, so on-board models **screen, detect, prioritise and compress**.
4. Part 2 covers how to build, run and trust such models on flight hardware.

---

# Part 2: Machine learning on the on-board computer

*Choosing, running and trusting models that decide what gets sent home.*

> **Where we left off:** the downlink is small, and the model decides what fits through it.
> Today: how to build that model, where it runs, and why you can trust it.

## 2.1 Four jobs for a model on board

| Job | Description | Mission example |
|---|---|---|
| **Screen** | drop unusable data | Φsat-1 detects clouds and discards heavily cloudy images |
| **Detect** | find what matters | Φsat-2 apps cover vessels, marine anomalies, wildfires |
| **Prioritise** | rank what is left so the most useful data goes first when contact time runs out | |
| **Compress** | shrink what is sent | Φsat-2 compresses images on board; they are reconstructed on the ground |

> **Design principle: choose by the cost of a mistake.**
> A false alarm wastes bytes; a missed event is gone for good.

> **Background: precision vs recall.** For a "discard" decision:
> - **Precision** = of everything the model discarded, how much really was junk.
> - **Recall** = of all the junk, how much the model caught.
>
> Because discarding good data cannot be undone, an on-board screen is tuned to be **cautious about
> discarding**: send whatever it is unsure about, and accept some wasted bytes. That is the demo's
> "unsure tiles are sent" policy.

## 2.2 From data to a model in orbit

```
01 Collect ──► 02 Train ──► 03 Shrink ──► 04 Test ──► 05 Monitor
                                                        │
                                   update or roll back ◄┘
```

| Step | What it means |
|---|---|
| **Collect** | labelled examples from earlier missions, labs or simulation |
| **Train** | on the ground, with ample compute |
| **Shrink** | fewer parameters, lower precision, simpler features |
| **Test** | on the target hardware, then uplink |
| **Monitor** | compare with downlinked samples, then update or roll back |

> **Training is not the hard part.** Getting *representative labelled data* and *checking the model
> on the real board* are.

> **Background: "lower precision"** usually means **quantisation**: storing weights as 8-bit
> integers instead of 32-bit floats. That cuts memory 4× and often speeds up inference, at a small
> cost in accuracy.

## 2.3 Match the model to the board

| Model class | Size | Strengths | Weaknesses |
|---|---|---|---|
| **Rules and thresholds** | bytes of code | easy to explain and verify | brittle at the margins |
| **Small models on features** (trees, tiny networks) | demo network: 75 parameters = **300 bytes** | cheap to run and to protect | need good hand-made features |
| **Deep networks on images** | large | most capable | need an accelerator or capable SoC; larger to store, test and protect |

Worked numbers: 75 parameters × 4 bytes (32-bit float) = 300 bytes. **Three copies** (for triple
redundancy, see 2.6) = **900 bytes**. Small models are cheap to protect.

## 2.4 Where the model can run

| Hardware | Characteristics | Example |
|---|---|---|
| **Microcontroller** (Cortex-M class) | low power, small memory; fits rules and tiny models | the Arduino demo |
| **Radiation-hardened CPU** | very reliable, but slow and costly for the compute it offers | RAD750, LEON |
| **SoC with FPGA** | general CPU plus reconfigurable logic | OPS-SAT: Cyclone V, dual-core ARM Cortex-A9 + FPGA, Linux |
| **Vision processing unit** | accelerator built for neural networks | Φsat-1: Intel Movidius Myriad 2 runs the cloud-detection network |

**Trade-offs:** compute, power, memory, radiation tolerance, cost, software tools.

## 2.5 Radiation and silent errors

**Single-event upset (SEU):** a charged particle flips a bit in memory or a register. No crash, no
warning.

Where it can hit a model:

- the input data;
- the weights or thresholds;
- the code;
- the decision log.

> A crash is easy to notice. A corrupted weight that quietly discards good data is not.

## 2.6 Standard protections

| Protection | How it works |
|---|---|
| **ECC memory** | detects and corrects flipped bits |
| **Scrubbing** | periodically re-reads and repairs stored weights |
| **Triple redundancy** | three copies, majority vote |
| **Watchdog** | restarts the processor if it hangs |
| **Plausibility checks** | reject inputs and outputs outside physical limits |
| **Confidence fallback** | if unsure, send the data instead of discarding it |

> **Background: why majority vote works.** With three copies, a single flipped bit corrupts only one
> copy, and the other two out-vote it. Failure needs two independent upsets hitting the *same* value
> before scrubbing repairs it, which is far less likely. ECC uses extra check bits in each memory
> word to the same end: typically it corrects any single-bit error and detects double-bit errors.

## 2.7 How do we know it is not discarding the good data?

| When | Practice |
|---|---|
| **Before launch** | report **recall and precision** on data that looks like flight data; test on the target board |
| **In flight** | downlink a **small random sample of discarded data** and compare it with the model's decisions |
| **Always** | **log decisions** and keep a rule that **errs toward sending** |

> **Watch for drift:** a sensor that ages, a new season or a storm can look nothing like the
> training data.

> **Background: why the random sample matters.** Without it you only ever see what the model chose
> to send, so its mistakes (good data discarded) are invisible by construction. A small random audit
> of the discards is the only direct estimate of the "red tile" rate in flight.

## 2.8 Models can change after launch

- **Φsat-2:** marine anomaly detection and wildfire detection were **uploaded after launch**.
- **OPS-SAT:** experimenters uploaded bootable images, with pre-checks before they ran.

Good practice:

- keep the previous model so you can **roll back**;
- **authenticate every uplink** and check it before use;
- upload a **small model, not the whole image**, when the uplink is slow.

## 2.9 Open questions (research directions)

| Area | Question |
|---|---|
| **Data for training** | Space events are rare and labels are scarce. How do we build trustworthy training sets? |
| **Validation** | How much evidence is enough before a model may discard data on its own? |
| **Efficiency** | What is the best accuracy per milliwatt and per kilobyte? |
| **Updates and drift** | How do we detect drift and retrain safely with a slow uplink? |
| **Radiation tolerance of learned models** | Which parts of a network matter most, and which need protection? |
| **Many satellites** | Can constellations share what they learn without sending raw data? |

## 2.10 If you want to try this

1. Pick one data-reduction job and **measure what a rule achieves first** (the baseline).
2. Train a small model and run it on a hobby board, as in the demo.
3. Add protections and a fallback, and **test with injected bit flips**.
4. Move to flight-like hardware and real or lab data.

---

## Discussion questions from the lecture

- If you could send only a handful of events on each pass, which ones would you send?
- Who do you trust: the satellite or the ground? And what would make you trust it?
- What would you never throw away? (Slide examples: wildfire monitoring, crop monitoring, ship
  tracking.)
- What would you let a satellite throw away? And what evidence would you need first?

## Key takeaways

1. **The link, not the sensor, is the limit.** A Sentinel-2 satellite generates 2.7 TB/day and only
   about 37% of it reaches the ground.
2. **On-board AI does four jobs:** screen, detect, prioritise, compress.
3. **Errors are asymmetric.** Wasted bytes are recoverable; discarded data is gone for good. Design
   for that: when unsure, send.
4. **Fit the model to the board.** Often a rule or a 300-byte network is the right answer, and small
   models are cheap to protect.
5. **Radiation causes silent errors.** Protect with ECC, scrubbing, triple redundancy, watchdogs,
   plausibility checks and a confidence fallback.
6. **Trust has to be earned and kept:** test before launch, audit random discards in flight, log
   decisions, watch for drift, keep a rollback.

## Glossary

| Term | Meaning |
|---|---|
| **CubeSat / U** | small standardised satellite built from 10 cm cube units (6U = six units) |
| **LEO** | low Earth orbit |
| **Pass** | the window when a satellite is in view of a ground station |
| **Downlink / uplink** | satellite → ground / ground → satellite |
| **OBC** | on-board computer |
| **Payload / housekeeping data** | mission measurements / satellite health telemetry |
| **ECV** | Essential Climate Variable |
| **S-band** | a radio band used for satellite links (up to 2 Mbps in the slide's example) |
| **NDVI** | Normalised Difference Vegetation Index, (NIR − Red)/(NIR + Red) |
| **SoC / FPGA** | system-on-chip / field-programmable gate array (reconfigurable logic) |
| **VPU** | vision processing unit, a neural-network accelerator (e.g. Movidius Myriad 2) |
| **SEU** | single-event upset: a radiation-induced bit flip |
| **ECC** | error-correcting code memory |
| **Scrubbing** | periodically re-reading and repairing stored data |
| **TMR** | triple modular redundancy: three copies and a majority vote |
| **Watchdog** | timer that restarts a hung processor |
| **Drift** | input data moving away from what the model was trained on |

## References (from the slides)

- ESA. OPS-SAT mission pages.
- ESA. Φsat-2 mission pages.
- Wikipedia. Phi-Sat-1.
- ESA. Climate Change Initiative (Essential Climate Variables) and Earth observation pages.
- CIRES, University of Colorado Boulder. Article on the February 2022 Starlink loss.
- OrbitChain. arXiv:2508.13374 (preprint, not peer reviewed): Sentinel-2 data volume and downlink
  figures.
- Radiation effects and mitigation (ECC, scrubbing, redundancy): general engineering practice; the
  speaker suggests adding a textbook or a NASA/ESA guideline of your choice.
- Demo kit: <https://github.com/Sudeshna2499/Fall-Campus-2026-Cloud-Detection-Demo>
