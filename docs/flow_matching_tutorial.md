# Flow matching in Plixer — a tutorial

*What `src/models/poc2mol_flow.py` actually computes, why each term is there, and what
follows from it. Assumes basic calculus (derivatives, the chain rule, "set the derivative to
zero to minimise") and nothing else.*

---

## 0. The whole thing in five lines

Everything in this document unpacks these five lines, which are `flow_loss()` in
`src/models/poc2mol_flow.py:289`:

```
x₀ ~ N(0, I)                     a grid of pure Gaussian noise
x₁ = the true ligand density      the answer we want to be able to produce
t  ~ p(t)  on [0, 1]              a random "how far along" number

xₜ = (1 − t)·x₀ + t·x₁            a point on the straight line from noise to ligand
u  = x₁ − x₀                      the direction that line travels in

loss = ‖ v_θ(xₜ, t, pocket) − u ‖²
```

In words: **take a real ligand, take some noise, stand at a random point on the straight line
between them, and ask the network which way to walk.** Train it with plain squared error.
That's it. There is no noise schedule, no ELBO, no adversarial game, no discrete diffusion
alphabet — one interpolation and one MSE.

The payoff is at the *end*: once the network can answer "which way to walk" from any point,
you can start at fresh noise and walk all the way to a ligand that was never in the training
set. That walk is what `sample()` does, and the ligand it produces is a **draw** from
p(ligand | pocket), not its average.

That distinction is the entire reason this branch exists, so let's build up to it properly.

---

## 1. What `x` is here

A ligand is stored as a **voxel grid**: a 3-D box of little cubes (32×32×32, each 0.75 Å on a
side), with 11 channels stacked on top — carbon, aromatic carbon, nitrogen, oxygen, sulphur,
halogens, and so on. Each number says "how much of this atom type is in this cube", between 0
(empty) and 1 (full).

So one ligand is an array of 11 × 32 × 32 × 32 = **360,448 numbers**, each in [0, 1].

Mathematically it doesn't matter that these are chemistry. Think of `x` as **one point in a
360,448-dimensional space**. A real ligand is one particular point. A different pose of the
same molecule is a *different* point, possibly quite far away. The set of all chemically
sensible ligand grids is some vanishingly thin, wildly curved surface inside that huge space —
"the data manifold".

**Model space.** Before anything else the occupancies are rescaled
(`to_model_space`, line 249):

```
x = (occupancy − 0.5) / 0.5          so [0, 1] ↦ [−1, +1]
```

Empty voxel → −1, full voxel → +1. Two reasons. First, the noise x₀ is standard Gaussian, which
lives around 0 with a spread of 1; if the data lived in [0, 1] the two ends of our line would
have wildly different scales and everything would be badly conditioned. Second — a fact we'll
use later — because ~96% of voxels are empty, almost every entry of x₁ is exactly −1, so the
**root-mean-square size of a real ligand grid in model space is ≈ 1**, the same as the noise.
Both ends of the journey have the same scale. That is a design choice, and it buys a free
diagnostic in §10.

---

## 2. The three ingredients

**x₀ — the noise.** `x₀ ~ N(0, I)` means: fill all 360,448 entries independently from a
standard normal (bell curve, mean 0, standard deviation 1). `I` is the identity matrix, which
here just means "no correlation between voxels" — pure static, no structure whatsoever.

**x₁ — the data.** One real ligand grid from HiQBind (or ZINC in stage A), in model space.

**t — the interpolation parameter.** A single number in [0, 1]. `t = 0` means "at the noise",
`t = 1` means "at the ligand". It is *not* wall-clock time and it is *not* a training step; it
is a coordinate along a path. Each training example gets its own random `t`.

⚠️ **Convention warning.** Half the literature uses the opposite convention (t = 0 is data,
t = 1 is noise — SD3 does this). This repo uses **t = 0 noise, t = 1 data**. Every formula
below is in *this* convention. This matters in §9.3, where the repo inherits a formula from a
paper that used the other one.

---

## 3. Perspective 1 — the straight line

### 3.1 The interpolation

```
xₜ = (1 − t)·x₀ + t·x₁
```

Read it as a weighted average that slides. At `t = 0` the weights are (1, 0) so `x₀ = x₀`. At
`t = 1` they are (0, 1) so `x₁ = x₁`. At `t = 0.3` you are 30% of the way from the noise to the
ligand — a recognisable but heavily static-corrupted ligand.

This is the **straight line segment** between two points in 360,448-dimensional space. Nothing
deeper. If you've ever written `lerp(a, b, t)` in graphics code, this is that.

### 3.2 The velocity — where the target comes from

Now differentiate the path with respect to `t`. Both x₀ and x₁ are fixed constants along one
path, so this is a one-line calculation:

```
d          d
── xₜ  =   ── [ (1 − t)·x₀ + t·x₁ ]  =  −x₀ + x₁  =  x₁ − x₀  =  u
dt         dt
```

**The training target `u` is literally the derivative of the path.** That's the whole reason
it's called a *velocity*: if you traverse the segment at a steady rate over one unit of `t`,
`x₁ − x₀` is your velocity vector — the direction you're heading, times how fast.

Two consequences fall straight out of the fact that the derivative has no `t` in it:

- **The velocity is constant along a given path.** Whether you're at `t = 0.1` or `t = 0.9` on
  *this particular* noise-to-ligand line, the correct answer is the same vector. The line is
  straight, so its direction never changes. This is what "rectified flow" means, and it's why
  the sampler can take big steps without much error.
- **The target does not depend on the network.** It is computable in closed form from the
  training pair. No simulation, no differential-equation solving inside the training loop —
  which is exactly what made earlier "continuous normalising flow" methods impractical, and
  what flow matching fixed.

### 3.3 The loss

```
                        1  N
ℒ(θ)  =  E            ─── Σ  ( v_θ(xₜ, t, pocket)ᵢ − uᵢ )²
          x₀,x₁,t       N  i=1
```

- `v_θ` is the U-Net. `θ` are its ~119M weights. It eats the noisy grid `xₜ`, the number `t`,
  and the pocket grid, and outputs a grid of the *same shape as the ligand* — a velocity vector
  for every voxel and channel.
- `Σᵢ (…)²` sums the squared error over all N = 360,448 entries; dividing by N makes it a mean.
  In code: `(velocity - target).pow(2).mean(dim=(1,2,3,4))` (line 336).
- `E[·]` — the expectation — means "average over the randomness": over which ligand you drew,
  which noise you drew, and which `t` you drew. In practice you approximate it by averaging over
  a minibatch, which is what SGD does with every loss.

So it is ordinary least-squares regression. The only unusual thing is *what* it regresses onto,
and that's the next section.

---

## 4. The trick: regressing onto a random target

Here is the thing that looks wrong the first time you see it.

Fix a specific noisy grid `xₜ` and a specific `t`. Many *different* (x₀, x₁) pairs could have
produced that same `xₜ`. A slightly different ligand plus slightly different noise lands in the
same place. Each of those pairs has a **different** target `u = x₁ − x₀`.

So the network is being shown the same input with contradictory labels. Surely it can't learn
anything?

It can, and here is the exact thing it learns. Let `u` be a random variable and ask: which
single number `v` minimises the expected squared error `f(v) = E[(v − u)²]`? Expand:

```
f(v) = E[v² − 2vu + u²] = v² − 2v·E[u] + E[u²]
```

Differentiate with respect to `v` and set to zero — first-year calculus:

```
f′(v) = 2v − 2·E[u] = 0     ⟹     v* = E[u]
```

(And `f″(v) = 2 > 0`, so it's a minimum.) **Squared error is minimised by the mean.** Fitting a
network with MSE against a noisy target doesn't produce noise; it produces the *conditional
expectation* of the target given the input. So the trained network converges to

```
v_θ*(x, t, pocket)  =  E[ x₁ − x₀ | xₜ = x, t, pocket ]
```

read as: *"averaged over every (ligand, noise) pair that could have produced this exact noisy
grid at this exact time in this exact pocket, which way were they all heading?"*

This object has a name — the **marginal velocity field** — and the theorem behind flow matching
(Lipman et al. 2023) is that following it transports the noise distribution exactly onto the
data distribution. We're not proving that here; what matters is understanding why the averaging
inside it is *harmless*, when averaging is precisely what broke the regression model.

---

## 5. "But isn't that an average again?" — the crux

The regression Poc2Mol minimised BCE+Dice against the true grid, so its optimum was
`E[x₁ | pocket]` — the average of every ligand pose the pocket admits. Averaging poses gives a
blur, which is why the measurements in `CLAUDE.md` §1 look the way they do: the prediction is
equivalent to rotating the truth by ~27°, and past 60° it matches a *wrong* pose better than
the right one. That is what a smear does.

Flow matching also learns an average. Why is this one fine?

### 5.1 The river

Think of a river with a rock in it. The water splits and goes around both sides. At any *point*
in the river there is one well-defined water velocity. That single velocity field is not a
"blurry compromise" between the two routes — it's just what the water is doing *there*.

Drop two leaves in at slightly different upstream positions and they end up on opposite sides
of the rock. One velocity field, two completely different destinations, neither of them the
average of the two.

**Averaging the velocity at each point is not the same as averaging the destinations.** The flow
field averages in a place where it does no damage; the regression model averaged the
destinations, which is exactly where it does maximal damage.

### 5.2 One impossible average, split into many easy ones

The other framing. Regression asks a single, brutally hard question:

> Given only the pocket, what is the ligand?

There genuinely are many answers, so the least-squares-optimal reply is their mean — a ghost.

Flow matching asks a *sequence* of much easier questions:

> Given that you are already 80% of the way toward one particular ligand, which way next?

At `t = 0.8` the noisy grid `xₜ` already contains most of the answer. The set of ligands
consistent with it is tiny. The average over that tiny set is barely an average at all — it's
essentially a single answer, and it's sharp.

The hard question hasn't vanished; it's been **spread over the path**, and the ambiguity that
remains at each step is resolved by the one thing regression never had: the random draw of x₀.

### 5.3 The first step really *is* the blurry mean (and that's fine)

Worth stating because it clarifies so much. At `t = 0` the state is `xₜ = x₀`, pure noise,
which carries no information about the ligand. So the optimal velocity there is

```
v*(x₀, 0) = E[x₁ − x₀ | x₀] = E[x₁ | pocket] − x₀
                              └──────┬──────┘
                          the exact blurry conditional mean
                          the regression model outputs
```

**The very first step of sampling points straight at the regression model's answer.** The
difference is not the first step. The difference is that flow matching doesn't *stop* there: as
`t` grows, `xₜ` accumulates commitment, the conditional set narrows, and the field bends toward
one specific ligand. The regression model outputs the ghost and hands it to you. Flow matching
uses the ghost as a starting bearing and then walks off it.

### 5.4 Straight paths, curved trajectories

A subtlety that trips people up. Each *training* path is a straight line. But the *sampling*
trajectory is generally **curved**.

Why: many training lines pass through the same region of space heading in different directions.
The learned field at a point is their average, so it doesn't agree exactly with any one of them,
and following it takes you along a bent route. In the river picture: individual water molecules
follow the streamlines, not the straight line from where they entered to where they exit.

Consequences:

- The curvature is why `sample_steps` matters at all. If trajectories were perfectly straight,
  **one** Euler step would be exact.
- The straighter the field, the fewer steps you need. That's what "rectified" flow is buying,
  and why 25–50 steps here rather than the 1000 that early diffusion models needed.
- "Reflow" methods (not implemented here) straighten the field by retraining on the model's own
  (noise, sample) pairs, which is the standard route to few-step sampling if you ever want it.

---

## 6. Perspective 2 — moving a cloud of points

A third way to see it, useful for grasping why the theorem is even true.

Imagine you don't have one noise sample but a **cloud** of millions of them, filling space in a
Gaussian blob. Now let every point in the cloud move according to the velocity field, all
together, for `t` from 0 to 1. The blob deforms: it stretches, folds, splits, and by `t = 1` it
has been squeezed onto the thin sheet of realistic ligand grids.

The whole game of flow matching is: **find a velocity field that deforms the Gaussian blob into
the data blob.** Once you have it, sampling is just "drop one point in and follow the flow".

The reason the straight-line construction works is that the interpolation defines the
intermediate blobs for us. At time `t`, the blob is "the distribution of `(1−t)x₀ + t·x₁` over
all pairs" — a Gaussian at `t=0`, the data at `t=1`, and something in between elsewhere. The
theorem says the conditional average of the per-pair velocities is *exactly* a field that
transports each blob into the next. (The formal statement is the continuity equation
`∂ₜpₜ + ∇·(pₜ vₜ) = 0`, which is just conservation of probability mass: whatever flows out of a
region has to show up somewhere else. You don't need it to use the method.)

This picture also explains what "the model diverged" means physically, and why the watchdog in
§10.3 works: a healthy trajectory rides inside the blob the whole way. A broken one gets ejected
into empty space where the network has never seen anything and its outputs are extrapolation.

---

## 7. Perspective 3 — it's a denoiser wearing a disguise

If you know diffusion models, this makes flow matching feel familiar immediately.

At sampling time you always know `xₜ` and `t`. So if the network gives you `v`, you can algebra
your way to an implied clean ligand. Start from the two facts you have:

```
xₜ = (1 − t)x₀ + t·x₁            (the path)
v  = x₁ − x₀                     (the velocity)
```

Two linear equations, two unknowns (x₀ and x₁). Substitute `x₀ = x₁ − v` into the first:

```
xₜ = (1 − t)(x₁ − v) + t·x₁ = x₁ − (1 − t)v
```

and therefore

```
x̂₁ = xₜ + (1 − t)·v            the implied clean ligand
x̂₀ = xₜ − t·v                  the implied noise
```

So **predicting the velocity, predicting the clean ligand, and predicting the noise are the same
task in three costumes** — each is an invertible affine function of the others given `(xₜ, t)`.
Diffusion's "ε-prediction" and "x₀-prediction" parameterisations are these same two alternatives.

The choice between them is not cosmetic, though, because it changes what the MSE *weights*:

- Predicting the noise makes the loss easy near `t = 1` (the noise has almost been removed and
  is easy to identify) and hard near `t = 0`.
- Predicting the clean data does the reverse.
- **Velocity prediction sits between them and is well-conditioned at both ends**, which is a
  large part of why the rectified-flow parameterisation is what SD3 and Flux converged on.

Practical use: `x̂₁ = xₜ + (1−t)v` is the single most useful debugging quantity in a flow model.
Compute it mid-trajectory and look at it — it shows you what the model currently *thinks* it's
producing, long before the trajectory finishes. It is also, in essence, what
`val/restore/dice_t50` probes (line 621): put the model on the true path at `t = 0.5` and see
whether it can finish the job.

---

## 8. Sampling — turning the field into a molecule

Training gives a field. Sampling integrates it. This is `sample()` (line 354) and it is the
numerical solution of an ordinary differential equation:

```
dx/dt = v_θ(x, t, pocket),        x(0) = x₀ ~ N(0, I),      answer = x(1)
```

### 8.1 Euler

Chop [0, 1] into `n` steps of size `Δt = 1/n` and repeatedly take

```
x ← x + Δt · v_θ(x, t)
```

This is the definition of a derivative run backwards: if `dx/dt ≈ v`, then over a short interval
`Δt` the state changes by about `v·Δt`. It's exact only if `v` doesn't change over the step,
which — see §5.4 — isn't quite true, so you accumulate error of order `Δt` per step.

### 8.2 Heun (the default)

Euler uses the velocity at the *start* of the step for the whole step, which systematically
overshoots on a curve. Heun's method fixes that by looking ahead:

```
v₀ = v_θ(x, t)                    velocity here
x̃  = x + Δt·v₀                    provisional Euler step
v₁ = v_θ(x̃, t + Δt)               velocity where you'd land
x ← x + Δt · ½(v₀ + v₁)           actually step with the average
```

Use the average of the slope at both ends rather than just the start — the same idea as the
trapezoid rule for integration, and the error drops from order `Δt` to order `Δt²`. It costs two
network evaluations per step, so 25 Heun steps ≈ 50 forward passes. Usually a better deal than
50 Euler steps, since accuracy improves quadratically while cost only doubles.

### 8.3 Why sampling is where things break

This deserves emphasis because it's the failure mode this codebase is built to catch.

During training, the network **only ever sees points exactly on a true noise-to-ligand line**.
Every `xₜ` it is asked about is an honest interpolation of a real ligand.

During sampling, integration error means you drift slightly off the manifold of such points.
Now the network is being asked about a state it never trained on — its answer is extrapolation.
That answer nudges you further off. The error compounds, and a model whose training loss is
falling beautifully can produce total garbage. §10.3 is how you detect it.

---

## 9. The knobs, and what each does to the equations

### 9.1 `sigma_min` (default 0.0)

The general form in the code is

```
xₜ = (1 − (1 − σ)t)·x₀ + t·x₁            target  u = x₁ − (1 − σ)·x₀
```

At `t = 1` this gives `x₁ + σ·x₀` — a floor of leftover noise at the data end. Setting `σ = 0`
recovers the exact straight line and makes `t = 1` land exactly on the data. Everything above
assumed `σ = 0`; it's the standard choice and the config default. `σ > 0` exists because some of
the original derivations wanted a strictly positive density everywhere. You almost certainly
don't need it.

### 9.2 `time_sampling` — how `t` is drawn (default `logit_normal`)

The loss is an average over `t`, so *how you draw `t`* decides where the network spends its
capacity. Two options (line 281):

**Uniform.** `t ~ U(0,1)`. Every point of the path equally often. Simple, but wasteful: near
`t = 0` the state is nearly pure noise and the best possible answer is nearly the blurry mean —
irreducibly high loss, little to learn. Near `t = 1` the answer is nearly determined —
irreducibly low loss, little to learn. Both ends are cheap; the middle is where the model has to
decide *which* ligand it's making.

**Logit-normal.** Draw `n ~ N(0,1)` and squash it through the logistic sigmoid:

```
t = sigmoid(μ + σ·n) = 1 / (1 + e^(−(μ + σn)))
```

The sigmoid maps ℝ → (0,1) with an S-shape whose slope is steepest at 0 → 0.5, so the resulting
density on `t` **piles up in the middle and thins at both ends**. With μ = 0 it's symmetric about
`t = 0.5`; μ > 0 shifts the mass toward the data end, σ controls concentration. This is SD3's
recipe and it measurably beats uniform on images.

Watch the per-bucket losses (`train/t_bucket_0` … `_7`, line 343) to see this directly: they
split the batch by which eighth of [0,1] its `t` fell in. You should see a monotone-ish ramp,
high at low `t`, low at high `t`. If the *middle* buckets are flat and high while the ends
improve, the model is failing exactly where it matters.

### 9.3 `time_shift` — and a convention trap worth knowing about

```
t ← s·t / (1 + (s − 1)·t)
```

A smooth reparameterisation of [0,1] onto itself, fixing both endpoints (plug in `t=0` and
`t=1`). `s = 1` is the identity. Applied to both the training draw and the sampling grid
(lines 287 and 411) so training density and step density agree.

Concretely, for `t = 0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0`:

| `s` | shifted values |
|---|---|
| 0.5 | 0.000 0.053 0.176 0.333 0.538 0.818 1.000 |
| 1.0 | 0.000 0.100 0.300 0.500 0.700 0.900 1.000 |
| 3.0 | 0.000 0.250 0.562 0.750 0.875 0.964 1.000 |

**`s > 1` pushes every `t` upward, i.e. toward `t = 1`, which in this repo's convention is the
DATA end.** Training draws land nearer the data; sampling steps bunch up near the data and take
big coarse jumps out of the noise.

⚠️ Now the trap. SD3 introduced this formula to spend *more* of the path near **noise** — the
motivation quoted in `configs/model/poc2mol_flow.yaml:24` — but SD3 uses the opposite convention
(their `t = 1` is noise). This repo has `t = 1` = data. **The same formula therefore has the
opposite effect here.** If the intent is SD3's, the sweep values that deliver it are `s < 1`
(0.33, 0.5), not `s > 1` (2.0, 3.0).

Nothing is broken — `s` is a perfectly good knob in either direction, and which direction wins
is an empirical question for a 32³×11 signal. Just know which way you're pushing, and sweep
both sides: `0.33, 0.5, 1.0, 2.0, 3.0`.

### 9.4 `cond_dropout_prob` and classifier-free guidance

On 10% of training samples the pocket is zeroed out (line 324):

```
cond ← cond · keep,        keep ~ Bernoulli(0.9), one draw per sample
```

The same weights therefore learn **two** fields: a conditional one `v_cond` (pocket present)
and an unconditional one `v_uncond` (pocket blank, "make me any plausible ligand"). It's
per-sample rather than per-batch so neither branch is ever starved.

At sampling time you combine them (line 439):

```
v = v_uncond + w·(v_cond − v_uncond)
```

Unpack the algebra. `(v_cond − v_uncond)` is **the part of the velocity that the pocket is
responsible for** — the difference between "make a ligand" and "make a ligand *for this
pocket*". Then:

- `w = 0` → pure unconditional; the pocket is ignored entirely.
- `w = 1` → the terms telescope to exactly `v_cond`; guidance off. (Also half the cost, since
  the unconditional branch needn't be evaluated at all — the code checks for this at line 408.)
- `w > 1` → **extrapolation past the conditional field.** You take the pocket's contribution and
  amplify it, deliberately overshooting in the direction the pocket pulls.

`w > 1` is the direct knob for "commit to a pose", and it has no analogue in the regression
model. The cost is diversity: you're pushing every trajectory toward the region the pocket
prefers most, so independent draws become more similar to each other and eventually distort
(too much guidance saturates and produces artefacts, the same as over-guided image models).
There is a sweet spot, typically 1.5–3.0, and it is **free to find** — it's a sampling-time
parameter, so `poc2mol_flow_eval.py --guidance 1.0 1.5 2.0 3.0` sweeps it on one trained
checkpoint without retraining anything.

### 9.5 The two representation constants

`occupancy_shift: 0.5`, `occupancy_scale: 0.5` are the [0,1] → [−1,1] map from §1. Changing them
changes the relative scale of data and noise — if the data were much smaller than the noise, the
`t ≈ 0` region would be all noise and the model would learn almost nothing there. The default
matches the two scales, which is what you want.

### 9.6 EMA

Not part of the objective, but part of the model. Weights are tracked by an exponential moving
average, `shadow ← 0.999·shadow + 0.001·current` (line 786). Generative models are routinely
several points better evaluated on the EMA than on the live weights, because SGD bounces around
a minimum and the average sits nearer the middle of the basin. Validation runs on the EMA and
the saved `state_dict` **is** the EMA, so the metric a checkpoint is selected on is measured on
the weights that checkpoint actually contains.

---

## 10. What the equations buy you

### 10.1 Worked example: the empty channel that used to smear

`CLAUDE.md` §5 records that under BCE+Dice the model emitted 59 units of fluorine mass into
ligands **containing no fluorine at all**. Here's why, and why this objective can't do that.

*Under Dice*: for an all-zero target channel the intersection `Σ(pred · target)` is identically
zero no matter what you predict, so the Dice term is 0 with **exactly flat gradient**. Only BCE
pushes back, ~56× more weakly. The model was free to smear.

*Under flow matching*: an empty channel has occupancy 0 everywhere, so in model space
`x₁ = −1` everywhere. Along its path,

```
xₜ = (1 − t)·x₀ + t·(−1)          target  u = −1 − x₀
```

Rearranging the first equation, `x₀ = (xₜ + t)/(1 − t)`. **The noise is exactly recoverable from
the state**, so the target is fully determined — the model can in principle achieve zero loss on
this channel, and quadratic error keeps pushing until it does. There is no floor and no flat
region. Emitting mass into a channel the ligand leaves empty is penalised at full strength,
proportional to the square of the excess.

The general statement: **MSE over velocity has no per-channel averaging and no zero-gradient
hole.** Every channel, occupied or not, carries real gradient at every `t`.

### 10.2 The 1-D example you can verify by hand

This is the whole argument of §5, small enough to compute exactly. One voxel. The pocket admits
two poses with equal probability: **occupied** (x₁ = +1) or **empty** (x₁ = −1). Noise
`x₀ ~ N(0,1)`.

**Regression's optimum** is `E[x₁] = 0` — occupancy 0.5, a half-there ghost voxel, a molecule
that exists nowhere. It cannot do better; 0 genuinely minimises its loss.

**Flow matching's optimum** we can derive. Given `xₜ = z` at time `t`, Bayes' rule on the two
Gaussian likelihoods gives `E[x₁ | z] = tanh(zt/(1−t)²)`, and after substituting
`x₀ = (z − t·x₁)/(1 − t)` the optimal velocity works out to

```
                tanh( z·t / (1−t)² ) − z
v*(z, t)  =  ─────────────────────────────
                        1 − t
```

Evaluate it:

| `t` | z = −0.5 | z = −0.1 | z = +0.1 | z = +0.5 |
|---|---|---|---|---|
| 0.0 | +0.500 | +0.100 | −0.100 | −0.500 |
| 0.1 | +0.487 | +0.097 | −0.097 | −0.487 |
| 0.3 | +0.290 | +0.056 | −0.056 | −0.290 |
| 0.5 | −0.523 | −0.195 | +0.195 | +0.523 |
| 0.7 | −1.664 | −1.838 | +1.838 | +1.664 |
| 0.9 | −5.000 | −9.000 | +9.000 | +5.000 |

Read the signs. **Early (t ≤ 0.3) the field points back toward 0** — toward the ghost, exactly
as §5.3 predicted. **From t ≈ 0.5 it flips** and drives away from 0, and by `t = 0.9` it is
driving hard toward ±1.

Integrate it and every trajectory commits:

```
x₀ = −1.20  →  −1.0000          x₀ = +0.05  →  +1.0000
x₀ = −0.50  →  −1.0000          x₀ = +0.50  →  +1.0000
x₀ = −0.05  →  −1.0000          x₀ = +1.20  →  +1.0000
```

The endpoint is **always ±1, never 0**. The sign of the initial noise decides which, and since
`x₀` is symmetric, exactly half the draws go each way — reproducing the true 50/50 distribution.

That is the whole thesis of this branch in six numbers: same data, same ambiguity, one objective
returns the average of the answers and the other returns the answers.

### 10.3 A free divergence alarm, from a two-line derivation

§1 arranged for `x₀` and `x₁` to have RMS ≈ 1. Compute the RMS along the path. Since `x₀` and
`x₁` are independent and `E[x₀] = 0`, the cross term vanishes:

```
E[xₜ²] = (1−t)²·E[x₀²] + 2t(1−t)·E[x₀]E[x₁] + t²·E[x₁²]
       = (1−t)² + 0 + t²
```

| t | 0.00 | 0.25 | 0.50 | 0.75 | 1.00 |
|---|---|---|---|---|---|
| RMS | 1.000 | 0.791 | **0.707** | 0.791 | 1.000 |

So **the RMS of a healthy state never leaves [0.707, 1]**, dipping to `1/√2` at the midpoint and
returning to 1 at both ends. Crucially this holds *regardless of how well the model has been
trained* — it's a property of the probability path, not of the weights.

That makes `val/sample/traj_rms_max` a **threshold-free alarm**. Whatever the model knows, if
the trajectory is healthy that number is ≈ 1. Over 2 means drifting; over 10 means diverging.
Compare with `val/sample/dice`, which cannot distinguish these cases at all: an untrained model
and a catastrophically diverged one both score ≈ 0.

The companion probe is `val/restore/dice_t50` (line 621): start from the **true** path at
`t = 0.5` and integrate the rest of the way with the same weights and sampler. It assigns blame.

| restore | sample | reading |
|---|---|---|
| high | low | The velocity field is fine; the **trajectory** is diverging. More steps, Heun, lower guidance, then `sample_clamp`. |
| low | low | The field isn't learned yet. Keep training — the sampler is not the problem. |
| low | high | Shouldn't happen. Suspect a metric bug. |

Stage A at step 10k measured restore 0.928 against sample 0.087: a near-perfect field near the
data, and a full trajectory from noise that hasn't come together yet. Textbook "keep training".

### 10.4 What the loss number does and doesn't tell you

`val/loss` here is the flow MSE. It is **not** comparable to the regression model's `val/loss`
(different objective entirely), and it is not the selection metric.

A large part of the flow MSE is **irreducible**. At low `t` the target `u = x₁ − x₀` contains a
large component of `x₀` that the state genuinely doesn't determine, so even a perfect model has
positive loss there. That floor doesn't move with model quality, so it dilutes the signal — the
same structural problem as the Dice floor, though for a different reason and to a much smaller
degree. And more fundamentally, a model can predict velocities well *near the data* while its
trajectory from noise still fails, which is precisely the §8.3 failure mode.

Select on **`val/sample/dice`** — integrate the ODE, score the result with the same pooled soft
Dice as the rotation control. **The number to beat is 0.596.** Read it next to
`val/sample/emission_ratio` (1.0 = calibrated) and the §10.3 watchdog. For unconditional stage A
the reference point is different: a perfect unconditional sample is simply a *different*
molecule, so the target is `dice(true_i, true_j)` between two real molecules — **0.2779** on the
ZINC validation split.

---

## 11. Where each formula lives

| formula | file:line |
|---|---|
| `xₜ = (1 − (1−σ)t)x₀ + t·x₁` | `poc2mol_flow.py:332` |
| `u = x₁ − (1−σ)x₀` | `:333` |
| `loss = mean((v − u)²)` | `:336` |
| `t = sigmoid(μ + σn)` | `:281` |
| `t ← s·t/(1 + (s−1)t)` | `:114` |
| pocket dropout | `:324` |
| occupancy ↔ model space | `:249`, `:253` |
| Euler / Heun steps | `:448`, `:450` |
| `v = v_uncond + w(v_cond − v_uncond)` | `:439` |
| trajectory RMS alarm | `:458`, `:473` |
| restoration probe | `:621` |
| EMA update | `:786` |
| how `t` enters the U-Net (sinusoidal features → MLP → per-channel FiLM scale/shift inside every residual block, zero-initialised so it starts as the identity) | `flow_unet3d.py:49`, `:86` |

---

## 12. Traps and FAQ

**"Is `t` the training step?"** No. It's a coordinate on the noise→ligand path, drawn afresh and
independently for every single training example. One minibatch contains many different `t`s.

**"Why not just train the network to output the ligand directly from noise?"** That's a GAN or a
one-shot regressor. Feeding it noise and asking for MSE against the ligand gives you back
`E[x₁ | pocket]` — the blur — because the noise is uninformative and MSE returns the mean (§4).
The path is what makes the problem well-posed at every point.

**"Does more `sample_steps` always help?"** It reduces integration error, with diminishing
returns, and it costs linearly. If `restore` is high and `sample` is low, more steps is the first
thing to try. If both are low, more steps buys nothing — you'd be integrating a field the model
hasn't learned.

**"Why is `sample_clamp` off by default?"** Because clamping makes a diverging model return a
plausible-*looking* saturated grid, and every diagnostic then understates the problem — `x = 10⁶`
and `x = 1.5` clamp to the same thing. Diagnose first, then enable it (2.0 is sensible).

**"Can I still get a deterministic single prediction?"** Yes — fix the seed for `x₀`. Same noise
plus same pocket gives a bit-identical sample, because the ODE is deterministic. All the
randomness lives in `x₀`. This is also how you compare settings fairly: pass the same `x0` and
change only guidance or step count.

**"Why does validation use fixed noise and a stratified `t` grid?"** So `val/loss` is the *same
quantity* every epoch and across runs (line 538). A stochastic validation loss is a moving
target for model selection, and "best val X" over a stochastic val is inflated by
maximum-selection bias.

**"Where does the pocket actually enter?"** It's concatenated onto the U-Net input channels —
the network sees the noisy ligand grid and the pocket grid side by side at full spatial
resolution, which is why they must be voxelised on the same grid. `t` enters separately, through
FiLM modulation inside every residual block.

---

## 13. Further reading

- Lipman et al., *Flow Matching for Generative Modeling* (2023) — the conditional flow matching
  theorem, i.e. why regressing the conditional velocity gives you the marginal field.
- Liu et al., *Rectified Flow* (2023) — the straight-line path and reflow.
- Esser et al., *Scaling Rectified Flow Transformers* / SD3 (2024) — logit-normal timestep
  sampling and the resolution shift. **Note their t-convention is the reverse of this repo's.**
- Ho & Salimans, *Classifier-Free Diffusion Guidance* (2022) — the `v_uncond + w(v_cond −
  v_uncond)` trick.
