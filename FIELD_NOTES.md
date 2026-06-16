# Build Small: Field Notes from "What Changed"

*A local, caregiver-operated Parkinson's diary that runs on a fine-tuned 1B model — built for the Build Small Hackathon (Backyard AI track).*

> **Not medical advice / not a medical device.** What Changed organizes a caregiver's own
> day-to-day observations into a report to share with a clinician. It does not diagnose,
> predict, or recommend treatment.

## The bet

The hackathon's constraint — ≤32B params — isn't a handicap to apologize for. It's a forcing
function. The most interesting place to point a *small* model is the data you'd never send to
a cloud API in the first place: someone's health, logged at home, every day. So our north star
became a tool where **"it runs locally, nothing leaves the device" is the whole pitch**, not a
checkbox.

## Finding a real problem (and killing our first idea)

We started from real pain, not a model demo. Using Reddit's API (PRAW, read-only) we read the
top posts across r/AgingParents, r/CaregiverSupport, r/dementia, and r/Alzheimers — hundreds of
threads written, overwhelmingly, by **adult children and spouses**, not the elders themselves.
That one observation reframed everything: *the user is the caregiver.*

Our first idea — a local "scam shield" for elderly people — died fast under an honest test:
the elder who won't open ChatGPT to vet a scam won't open *anything*, and a scam text isn't
private data, so "local" earned nothing. Rule we adopted: **the adopter is the tech-capable
caregiver, and local has to be load-bearing.**

That led to a caregiver decline-tracker — and then to a sharper question: *generalize, or
specialize?* We specialized to **Parkinson's disease**, because specificity is exactly what the
problem (and the judges) reward, and because PD is almost custom-made for a home diary.

## Grounding it in real clinical tools

We didn't invent a schema from vibes. We anchored to the two instruments every
movement-disorder neurologist knows:

- The **Hauser PD Home Diary** — the validated ON/OFF diary that tracks *OFF time* (meds
  wearing off) and *troublesome dyskinesia*, so doses can be re-timed.
- The **MDS-UPDRS** — the gold-standard scale (motor daily-living items like tremor, freezing,
  walking; non-motor items like mood, sleep, cognition, hallucinations).

So What Changed is, in effect, a **simplified daily Hauser-style ON/OFF diary + MDS-UPDRS-aligned
tracker**: six daily 1–5 ratings (mobility, tremor, stiffness, mood, sleep, alertness) and six
event counts (OFF episodes, dyskinesia spells, freezing, falls, missed/late doses,
hallucinations).

## The architecture bet: keep the model off the hot path

The single most important design decision: **the trend detection is deterministic Python, and
the LLM only does the boring parts.** Logging and trend math (rolling averages, decline streaks,
event rates, the "OFF episodes creeping earlier" signal) never touch a model — so they never
hallucinate. The fine-tuned model does exactly two narrow jobs: turn an optional free-text note
into structured fields, and narrate the already-computed findings in plain language.

This is why a *1B* model is enough. We're not asking it to reason about Parkinson's; we're asking
it to read "froze in the doorway, pill wore off before lunch" and fill in a form. Narrow task →
tiny model → fully local.

## Building it — and the bug 30 green tests missed

The app was built test-first (SQLite store → trend engine → inference layer → Gradio UI), with
a fresh agent per task and a review after each. The lesson worth sharing: **our 30 passing unit
tests all missed a critical bug.** A final, holistic end-to-end review caught it — the app shared
one SQLite connection across Gradio's worker threads (`check_same_thread=True`), so *every button
click* would have crashed in the live app. Unit tests passed because they ran on the main thread.
Evidence over assertions: a review that actually exercises the whole flow earns its keep.

## The small model

The shipped model is a fine-tuned **MiniCPM5-1B** (~1.08B params, Apache-2.0, llama-architecture
→ clean LoRA + GGUF), run locally via llama.cpp. That single choice stacks the hackathon's
"build small" ethos into real badges: **Well-Tuned** (a published fine-tune), **Off the Grid**
(no cloud APIs at runtime), and **Llama Champion** (llama.cpp) — joined later by **Off-Brand**
for the hand-built interface.

## The data (and a humbling quality saga)

There's no public dataset for "Parkinson's caregiver note → JSON," and real caregiver notes are
exactly the PII we refuse to collect. So we generated synthetic data **label-first**: sample a
ground-truth JSON, then have a teacher paraphrase it into a natural caregiver note. Labels are
correct by construction.

It took several iterations to get clean data, and the failures were instructive:

1. **v1 taught the model *backwards*.** The teacher didn't know our convention (5 = best), so
   "appetite 5/5" became *"barely ate."* Fix: render ratings with valence words ("very good
   (5/5, where 5 is best)").
2. **The teacher parroted our examples.** Concrete phrases we put in the prompt got stamped onto
   notes — even onto *empty-label* days, which would have taught "symptoms present → nothing
   notable." Fix: specify *tone*, never copyable phrases.
3. **Then it went stiff.** Removing examples made it write clinical restatements. Fix: a
   tone-only prompt + an output cleaner (strip thinking blocks + preambles).

But there's a ceiling the prompt can't fix. Even after a weaker 8B teacher's data passed our
automated checks, *reading the eval notes by hand* surfaced two problems the metrics missed:
**a third of the notes were written *to* the patient** ("Mom, you barely slept, huh?") instead
of being a caregiver's third-person diary, and **~1% silently swapped freezing-of-gait for a
fall**. Tightening the prompt just traded one failure for another — that whack-a-mole is the
tell that the *model*, not the prompt, is the bottleneck.

So we swapped the teacher: **Claude Sonnet 4.6** regenerated all the notes under a strict
contract — third person, never invert a symptom's direction, counts in words, plain language,
no jargon. We validated every batch both mechanically (voice, digits, event-concept coverage,
freezing/fall swaps) *and* by reading samples. The fine-tune jumped from **0.81 to 0.97**
exact-match field-F1 — entirely from data quality.

Takeaway: with synthetic data, your prompt *is* your dataset — read your samples; and once the
prompt is debugged and it still slips, upgrade the model, not the prompt.

## Measuring honestly: baseline before training

Before fine-tuning, we measured the **base MiniCPM5-1B zero-shot** on the eval set with the exact
production prompt + parser — on a Modal GPU, so the baseline and the post-training number are
apples-to-apples.

- Base (zero-shot) field-F1: **~0.00 exact / ~0.08 within-±1** — degenerate; the base can't do
  the task untrained.
- Fine-tuned (LoRA, 2 epochs): **0.97 exact / 0.99 within-±1** in-distribution; **0.83 / 0.91**
  on an adversarial OOD probe. Same prompt/parser at train, eval, and serve.

(Training: LoRA via Unsloth on an A10G — ~0.44 s/step, **~66 s** end to end. The leap from a
first fine-tune at 0.81/0.91 came entirely from data quality, not hyperparameters.)

## Shipping a 1B to a free CPU box (the deploy is part of "small")

A local model is only "local" if it actually loads on the target, and getting our GGUF to run
on Hugging Face's free **CPU** Space was its own saga. The file was written by Unsloth's
bleeding-edge llama.cpp, so it needed a recent **llama-cpp-python (≥ 0.3.26)** to parse — but
every prebuilt wheel was either **musl** (wrong libc for the glibc Space) or too old to read it.
So we built a **glibc 0.3.26 wheel on Modal** (with a `-pthread` link fix), *verified it loads
the GGUF before shipping*, hosted it on the model repo, and pinned the Space to it.

Then a trap that's easy to miss: on a 2-vCPU container, llama.cpp reads the *host's* core count
and spawns far too many threads — one extraction took ~46s of thrashing. Pinning `n_threads` to
the real vCPU count and shrinking `n_ctx` brought it back to single digits. **Lesson: "build
small" includes the deploy. The win condition isn't the eval number; it's the model loading and
responding on the free tier the user actually has.**

## Make the model the hero: a note-first UI

Our first layout led with sliders and treated the note as optional — backwards, since the
note→structure extraction *is* the product. We flipped it: the **note is the primary input**
with the fine-tuned auto-fill as the hero action, and the sliders became a review-and-correct
surface. Each 1–5 slider got **plain-language anchors** ("1 = barely slept … 5 = slept very
well") so a caregiver rates the day's *quality* instead of guessing what "3" means, and the
free-text date became a **Y/M/D picker**. Small moves, but now the model's contribution is the
first thing you see.

## The chart is an argument

The trend view began as one chart with six overlapping lines — unreadable, and the *events*
(falls, OFF) weren't plotted at all. Rebuilding it taught three things:

- **Small multiples beat a tangle.** Six smoothed mini-charts each read clearly, and a **7-day
  rolling average** turns daily jitter into a trend — so a noisy-but-stable signal like sleep
  correctly reads as *flat*, not as alarming swings.
- **Don't color a trend you can't defend.** We first colored event bars by direction — until the
  data pushed back: missed-meds flickered "improving" on 1↔2 noise, and dyskinesia "appeared,
  peaked, then receded," which no single color can honestly express. So events are **neutral
  bars** and the shape tells the story; only the smoothed ratings get red/grey/green.
- **Encoding follows data type.** Ratings are a standing level → **lines**; events are discrete
  counts → **bars** — the visual form of the same rule we keep in the schema: severity and
  counts are different things on different scales.

## Off the hot path, again: the report

Generating the report once appeared to "do nothing" for half a minute — the deterministic
content was instant, but it blocked on a 256-token LLM narration on CPU. Fix: render the
**deterministic summary immediately**, and make the AI-written paragraph an **opt-in** that
**falls back to the factual summary** if it's slow or errors. Same principle as the trend
engine: the model garnishes the report, it isn't the engine.

## Voice notes, still 100% local

The most natural way for a tired caregiver to log a day is to *say* it. We added on-device
speech-to-text with **faster-whisper** (Whisper on CTranslate2), which ships as clean prebuilt
wheels — no compiler, so it didn't reopen the build saga — and keeps the privacy story whole:
**speak → transcribed on the device → extracted by the 1B → trends + report**, with nothing
leaving the machine.

## What "build small" taught us

A specialized 1B model that runs on the caregiver's own machine beats reaching for a frontier
API here on every axis that matters: privacy, offline, cost, and *trust*. The constraint didn't
limit the product — it pointed us at the version of the product worth building.

— Built with [Gradio](https://gradio.app) + [Hugging Face](https://huggingface.co), MiniCPM
(OpenBMB), faster-whisper, and Modal.
