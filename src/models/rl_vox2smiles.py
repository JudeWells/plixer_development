"""GRPO and DPO fine-tuning of the Vox2Smiles decoder against a Tanimoto reward.

Everything before this point trained the decoder by maximum likelihood on the ONE true ligand
per pocket. That objective cannot express "this molecule is nearly right": a generation that
differs from the reference by a single methyl is scored exactly as wrong as benzene. Tanimoto
similarity can, so this module optimises it directly -- sample molecules from the policy for a
pocket, score each against the true ligand, and push the policy towards the better ones.

    pocket --[frozen Poc2Mol]--> ligand density --[decoder = policy]--> sampled SMILES
                                                                            |
                                                        Morgan-Tanimoto to the true ligand
                                                                            |
                                                              GRPO advantage / DPO preference

Two objectives, selected by ``objective``:

``grpo``  Sample ``group_size`` completions per pocket, standardise the rewards WITHIN each
          pocket's group, and use that as the advantage on the sequence log-probability. The
          within-group baseline is the whole point: pockets differ enormously in how well any
          model can do on them, so a global baseline would mostly reward easy pockets rather
          than good generations. With one gradient step per sampling round the PPO ratio is
          identically 1, so no clipping term is needed and this reduces to REINFORCE with a
          group baseline -- which is what GRPO is at mu = 1.
``dpo``   From the same group, take the best and worst by reward as a preference pair and
          apply the usual DPO loss against a frozen reference. Pairs whose reward gap is below
          ``dpo_min_margin`` are dropped: at small gaps the "preference" is sampling noise, and
          training on it teaches the model to distinguish molecules that are equally good.

Both are anchored to a frozen copy of the starting policy -- GRPO through an explicit KL
penalty, DPO through the reference terms in its loss. Without that anchor a Tanimoto reward is
trivially hackable: CLAUDE.md records the decoder already emits a near-constant smear when
under-constrained, and the highest-reward degenerate policy is to ignore the pocket and always
emit whatever scaffold is most common in HiQBind.

⚠️ THE REWARD AND THE PROJECT METRIC ARE DIFFERENT THINGS. `val/likelihood_auc_znorm` is a
*ranking* metric computed from teacher-forced likelihoods; this optimises *generation*. RL that
sharpens a policy usually costs calibration, so the AUC can fall while the reward rises. Both
are logged every validation. Read them together and do not report one without the other.
"""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F

from src.models.end_to_end import EndToEndPoc2Smiles
from src.utils.metrics import paired_similarities


class RLVox2Smiles(EndToEndPoc2Smiles):
    """Policy-gradient / preference fine-tuning of the decoder. Poc2Mol stays frozen.

    Subclasses ``EndToEndPoc2Smiles`` purely to inherit the pipeline plumbing -- it already
    knows how to turn a pocket batch into decoder ``pixel_values`` via a frozen Poc2Mol, and it
    already computes `val/likelihood_auc_znorm` by the same code path the 0.7589 baseline came
    from. The upstream is hard-frozen here: the end-to-end gradient was measured at −0.0024
    (paired, 8 seeds, n.s.) while multiplying run-to-run variance 7×, and the user's request is
    explicitly RL "on the language model".

    Args:
        objective: ``"grpo"`` or ``"dpo"``.
        group_size: completions sampled per pocket. GRPO's baseline is the mean over this
            group, so it must be >= 2; below ~4 the standardised advantage is very noisy.
        sample_temperature / top_p: sampling for the rollout. Too low and every member of the
            group is the same molecule, the group std collapses and the advantage is 0/0.
        kl_beta: weight of the KL-to-reference penalty (GRPO only; DPO carries its reference
            inside the loss).
        dpo_beta: the usual DPO inverse-temperature.
        dpo_min_margin: minimum reward gap for a pair to be used.
    """

    def __init__(
        self,
        config,
        poc2mol_model,
        objective: str = "grpo",
        group_size: int = 8,
        sample_temperature: float = 1.0,
        top_p: float = 0.95,
        max_generation_length: int = 200,
        kl_beta: float = 0.02,
        dpo_beta: float = 0.1,
        dpo_min_margin: float = 0.05,
        **kwargs,
    ) -> None:
        # The upstream plays no part in RL. Pinning these three here rather than trusting the
        # config means a stray override cannot quietly turn the experiment into a
        # simultaneous end-to-end run, which would confound the only thing being measured.
        kwargs["voxel_loss_weight"] = 0.0
        kwargs["poc2mol_lr"] = 0.0
        kwargs["lm_grad_to_poc2mol"] = False
        super().__init__(config, poc2mol_model, **kwargs)

        if objective not in {"grpo", "dpo"}:
            raise ValueError(f"objective must be 'grpo' or 'dpo', got {objective!r}")
        if group_size < 2:
            raise ValueError("group_size must be >= 2; the baseline is the group mean")

        self.objective = objective
        self.group_size = int(group_size)
        self.sample_temperature = float(sample_temperature)
        self.top_p = float(top_p)
        self.max_generation_length = int(max_generation_length)
        self.kl_beta = float(kl_beta)
        self.dpo_beta = float(dpo_beta)
        self.dpo_min_margin = float(dpo_min_margin)

        # Frozen, and excluded from the optimiser and from DDP's reduction. Plain DDP treats a
        # parameter that receives no gradient as a hard error, so freezing is not merely an
        # optimisation here.
        for parameter in self.poc2mol.parameters():
            parameter.requires_grad_(False)

        # Held in a LIST so ``nn.Module.__setattr__`` does not register it as a child. If it
        # were a child it would be deep-copied into every checkpoint (+172M parameters), and
        # DDP would try to reduce it. Populated in ``on_fit_start`` -- see there for why it
        # cannot be built in __init__.
        self._reference = []

    # ------------------------------------------------------------------ reference policy

    def on_fit_start(self):
        """Snapshot the starting policy as the reference.

        ⚠️ This CANNOT be done in ``__init__``. ``src/train.py`` applies ``init_weights_from``
        *after* ``hydra.utils.instantiate``, so at construction time ``self.model`` still holds
        randomly-initialised weights. A reference copied then would be a random network, the KL
        penalty would be pulling the policy towards noise, and DPO's reference terms would be
        meaningless -- while everything still ran and logged plausible-looking curves. By
        ``on_fit_start`` the checkpoint is loaded and the module is on its device.
        """
        reference = copy.deepcopy(self.model).eval()
        for parameter in reference.parameters():
            parameter.requires_grad_(False)
        self._reference = [reference.to(self.device)]

    @property
    def reference(self):
        if not self._reference:
            raise RuntimeError("reference policy not built -- on_fit_start did not run")
        return self._reference[0]

    # ------------------------------------------------------------------ rollout

    @torch.no_grad()
    def sample_group(self, pixel_values):
        """Sample ``group_size`` completions for every row.

        Returns ``(repeated_pixel_values, tokens)`` with the rows interleaved, so row i of the
        input owns rows ``i*G .. (i+1)*G - 1`` of the output and a simple ``.view(B, G)``
        recovers the grouping.
        """
        repeated = pixel_values.repeat_interleave(self.group_size, dim=0)
        tokens = self.model.generate(
            repeated,
            do_sample=True,
            temperature=self.sample_temperature,
            top_p=self.top_p,
            max_length=self.max_generation_length,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        return repeated, tokens

    def token_logprobs(self, module, pixel_values, tokens):
        """Per-token log-probability of ``tokens`` under ``module``, with its validity mask.

        Teacher-forces on the sampled sequence explicitly -- ``decoder_input_ids = tokens[:, :-1]``
        predicting ``tokens[:, 1:]`` -- rather than going through the ``labels=`` path. The two
        are equivalent here (``tokens[:, 0]`` is the decoder start token that ``generate``
        emitted, which is exactly what HuggingFace's internal shift would prepend), but the
        explicit form makes the off-by-one impossible to get wrong, and it skips HF's redundant
        internal loss computation.
        """
        pad_id = self.tokenizer.pad_token_id
        attention = (tokens != pad_id).long()
        outputs = module(
            pixel_values=pixel_values,
            decoder_input_ids=tokens[:, :-1],
            decoder_attention_mask=attention[:, :-1],
            return_dict=True,
        )
        log_probs = torch.log_softmax(outputs.logits.float(), dim=-1)
        targets = tokens[:, 1:]
        token_lp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        # The first EOS is a real, predicted token and must be kept; only padding after it is
        # masked. Dropping EOS would leave the policy with no gradient on when to stop.
        mask = (targets != pad_id).float()
        return token_lp * mask, mask

    def rewards_for(self, tokens, true_smiles):
        """Morgan-Tanimoto of each sampled molecule against its own pocket's true ligand.

        Uses ``paired_similarities``, i.e. the exact quantity behind the ``val/*/tanimoto``
        metric -- including its convention that an unparseable generation scores 0.0 instead of
        being dropped, which is what stops the policy from farming reward by emitting garbage
        on pockets it finds hard.
        """
        decoded = self.tokenizer.batch_decode(tokens, skip_special_tokens=True)
        generated = [text.replace(" ", "") for text in decoded]
        references = [
            reference for reference in true_smiles for _ in range(self.group_size)
        ]
        scores = paired_similarities(generated, references)
        return torch.tensor(scores, device=tokens.device, dtype=torch.float32), generated

    # ------------------------------------------------------------------ objectives

    def grpo_loss(self, rewards, policy_lp, reference_lp, mask, batch_size):
        """Group-standardised REINFORCE with a KL leash."""
        lengths = mask.sum(-1).clamp(min=1.0)
        # Length-normalised, so a long molecule does not receive a systematically larger
        # gradient than a short one purely for having more tokens.
        sequence_lp = (policy_lp.sum(-1) / lengths).view(batch_size, self.group_size)

        grouped = rewards.view(batch_size, self.group_size)
        centred = grouped - grouped.mean(dim=1, keepdim=True)
        # A group whose samples all scored identically carries no preference information; the
        # +1e-6 leaves its advantage at 0 rather than dividing 0 by 0.
        advantage = centred / (grouped.std(dim=1, keepdim=True) + 1e-6)

        policy_gradient = -(advantage.detach() * sequence_lp).mean()

        # k3 estimator: unbiased, non-negative, and lower variance than the naive difference.
        delta = (reference_lp - policy_lp)
        kl_per_token = torch.exp(delta) - delta - 1.0
        kl = (kl_per_token * mask).sum() / mask.sum().clamp(min=1.0)

        return policy_gradient + self.kl_beta * kl, {"kl": kl.detach(),
                                                     "pg": policy_gradient.detach()}

    def dpo_loss(self, rewards, policy_lp, reference_lp, mask, batch_size):
        """Best-vs-worst preference pairs within each group."""
        policy_sum = (policy_lp.sum(-1)).view(batch_size, self.group_size)
        reference_sum = (reference_lp.sum(-1)).view(batch_size, self.group_size)
        grouped = rewards.view(batch_size, self.group_size)

        best = grouped.argmax(dim=1)
        worst = grouped.argmin(dim=1)
        rows = torch.arange(batch_size, device=rewards.device)
        margin = grouped[rows, best] - grouped[rows, worst]
        usable = margin >= self.dpo_min_margin

        chosen = policy_sum[rows, best] - reference_sum[rows, best]
        rejected = policy_sum[rows, worst] - reference_sum[rows, worst]
        logits = self.dpo_beta * (chosen - rejected)
        per_pair = -F.logsigmoid(logits)

        if not bool(usable.any()):
            # Keep every parameter in the graph with a zero gradient rather than returning a
            # constant: plain DDP rejects a parameter that received no gradient, and a batch
            # where every group is flat is not rare early on, when the policy is confident.
            return policy_sum.sum() * 0.0, {"pairs": torch.tensor(0.0, device=rewards.device),
                                            "margin": margin.mean().detach()}
        loss = per_pair[usable].mean()
        return loss, {"pairs": usable.float().sum().detach(),
                      "margin": margin[usable].mean().detach()}

    # ------------------------------------------------------------------ training

    def training_step(self, batch, batch_idx):
        prepared, _ = self._prepare(batch, training=True)
        pixel_values = prepared["pixel_values"]
        batch_size = pixel_values.size(0)

        true_smiles = [
            smiles.replace("[BOS]", "").replace("[EOS]", "")
            for smiles in batch["smiles_str"]
        ]

        repeated, tokens = self.sample_group(pixel_values)
        rewards, generated = self.rewards_for(tokens, true_smiles)

        policy_lp, mask = self.token_logprobs(self.model, repeated, tokens)
        with torch.no_grad():
            reference_lp, _ = self.token_logprobs(self.reference, repeated, tokens)

        if self.objective == "grpo":
            loss, extra = self.grpo_loss(rewards, policy_lp, reference_lp, mask, batch_size)
        else:
            loss, extra = self.dpo_loss(rewards, policy_lp, reference_lp, mask, batch_size)

        grouped = rewards.view(batch_size, self.group_size)
        n = batch_size * self.group_size
        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True, batch_size=n)
        # The mean is what the policy is pushed towards; the best-in-group is what sampling can
        # currently reach at all, and is the ceiling the mean can be dragged to. If best-in-group
        # is flat there is nothing for the advantage to select and the rollout needs more
        # temperature or a larger group.
        self.log("train/reward_mean", grouped.mean(), on_step=True, on_epoch=True,
                 prog_bar=True, batch_size=n)
        self.log("train/reward_best_in_group", grouped.max(dim=1).values.mean(),
                 on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("train/reward_std_in_group", grouped.std(dim=1).mean(),
                 on_step=True, on_epoch=True, batch_size=batch_size)
        self.log("train/valid_fraction",
                 torch.tensor(float(sum(1 for s in generated if s) / max(len(generated), 1)),
                              device=rewards.device),
                 on_step=True, on_epoch=True, batch_size=n)
        self.log("train/gen_tokens", mask.sum(-1).mean(), on_step=True, on_epoch=False,
                 batch_size=n)
        for name, value in extra.items():
            self.log(f"train/{name}", value, on_step=True, on_epoch=False, batch_size=n)
        return loss

    # ------------------------------------------------------------------ optimiser

    def configure_optimizers(self):
        """Decoder only. Poc2Mol is frozen, and the reference is not a child module."""
        weight_decay = float(getattr(self.hparams.config, "weight_decay", 0.01))
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            parameters, lr=self.hparams.config.lr, weight_decay=weight_decay
        )
        return self._attach_scheduler(optimizer)
