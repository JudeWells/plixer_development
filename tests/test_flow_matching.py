"""Checks for the generative (flow-matching) Poc2Mol.

pytest is not installed in venvPlixer, so every test is a plain function and the file is
runnable directly:

    ./venvPlixer/bin/python tests/test_flow_matching.py

The last test actually trains a tiny network for a few hundred steps and asserts that
samples move towards the target density. It is the only one that can catch a sign error in
the probability path -- shapes and finiteness cannot.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.flow_unet3d import TimeConditionedResidualUNetSE3D  # noqa: E402
from src.models.poc2mol_flow import (  # noqa: E402
    FlowUnetConfig,
    Poc2MolFlow,
    apply_time_shift,
    pooled_soft_dice,
)


def tiny_model(ligand_channels=2, protein_channels=1, **kwargs):
    config = FlowUnetConfig(
        ligand_channels=ligand_channels,
        protein_channels=protein_channels,
        f_maps=8,
        num_groups=4,
        num_levels=2,
        temb_dim=16,
    )
    return Poc2MolFlow(config, **kwargs)


def test_unet_shapes_and_zero_init():
    net = TimeConditionedResidualUNetSE3D(
        in_channels=3, out_channels=3, cond_channels=2, f_maps=8, num_groups=4, num_levels=2
    )
    x = torch.randn(2, 3, 8, 8, 8)
    cond = torch.randn(2, 2, 8, 8, 8)
    t = torch.rand(2)
    out = net(x, t, cond)
    assert out.shape == x.shape, out.shape
    # Zero-initialised final conv: the ODE starts as the identity map.
    assert torch.count_nonzero(out) == 0

    # Both the output head and the FiLM heads are zero-initialised, so at step zero the
    # network is deliberately constant. Wake them up before asking what it depends on.
    torch.nn.init.normal_(net.final_conv.weight, std=0.1)
    for module in net.modules():
        if hasattr(module, "film"):
            torch.nn.init.normal_(module.film.weight, std=0.1)
    a = net(x, torch.zeros(2), cond)
    b = net(x, torch.ones(2), cond)
    assert not torch.allclose(a, b), "output is independent of the timestep"

    # ...and on the condition.
    c = net(x, torch.ones(2), torch.zeros_like(cond))
    assert not torch.allclose(b, c), "output is independent of the condition"
    print("ok  unet shapes / zero init / t and cond dependence")


def test_unconditional_unet_rejects_condition():
    net = TimeConditionedResidualUNetSE3D(
        in_channels=2, out_channels=2, cond_channels=0, f_maps=8, num_groups=4, num_levels=2
    )
    out = net(torch.randn(1, 2, 8, 8, 8), torch.rand(1))
    assert out.shape == (1, 2, 8, 8, 8)
    try:
        net(torch.randn(1, 2, 8, 8, 8), torch.rand(1), torch.randn(1, 1, 8, 8, 8))
    except ValueError:
        print("ok  unconditional net rejects a condition")
        return
    raise AssertionError("unconditional net silently accepted a condition")


def test_time_shift():
    t = torch.linspace(0, 1, 11)
    assert torch.allclose(apply_time_shift(t, 1.0), t)
    shifted = apply_time_shift(t, 3.0)
    assert torch.isclose(shifted[0], torch.tensor(0.0))
    assert torch.isclose(shifted[-1], torch.tensor(1.0))
    assert (shifted.diff() > 0).all(), "shift must stay monotone"
    # shift > 1 moves mass towards t = 1, i.e. every interior point rises
    assert (shifted[1:-1] > t[1:-1]).all()
    print("ok  time shift endpoints / monotonicity")


def test_occupancy_roundtrip():
    model = tiny_model()
    occ = torch.rand(2, 2, 4, 4, 4)
    assert torch.allclose(model.to_occupancy(model.to_model_space(occ)), occ, atol=1e-6)
    # The default mapping puts occupancy on [-1, 1].
    assert torch.isclose(model.to_model_space(torch.zeros(1)), torch.tensor(-1.0))
    assert torch.isclose(model.to_model_space(torch.ones(1)), torch.tensor(1.0))
    print("ok  occupancy round trip")


def test_flow_loss_and_gradients():
    model = tiny_model()
    ligand = torch.rand(3, 2, 8, 8, 8)
    protein = torch.rand(3, 1, 8, 8, 8)
    out = model.flow_loss(ligand, protein, bucket_diagnostics=True)
    assert torch.isfinite(out["loss"])
    assert any(k.startswith("t_bucket_") for k in out)
    out["loss"].backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no gradients reached the parameters"
    assert any(g.abs().sum() > 0 for g in grads), "all gradients are exactly zero"

    # Channel-count mismatches must fail loudly rather than broadcast.
    try:
        model.flow_loss(torch.rand(3, 5, 8, 8, 8), protein)
    except ValueError:
        pass
    else:
        raise AssertionError("wrong ligand channel count was accepted")
    print("ok  flow loss finite, differentiable, shape-checked")


def test_sampling_determinism_and_guidance():
    model = tiny_model().eval()
    torch.nn.init.normal_(model.model.final_conv.weight, std=0.05)
    protein = torch.rand(2, 1, 8, 8, 8)

    def draw(seed, **kwargs):
        generator = torch.Generator().manual_seed(seed)
        return model.sample(protein=protein, n_steps=4, generator=generator, **kwargs)

    a, b = draw(0), draw(0)
    assert torch.equal(a, b), "sampling is not reproducible under a fixed generator"
    assert not torch.equal(a, draw(1)), "different seeds gave identical samples"
    assert a.shape == (2, 2, 8, 8, 8)
    assert (a >= 0).all() and (a <= 1).all(), "occupancies escaped [0, 1]"

    guided = draw(0, guidance_scale=3.0)
    assert not torch.equal(a, guided), "guidance_scale had no effect"

    euler = draw(0, sampler="euler")
    assert not torch.equal(a, euler), "heun and euler produced identical trajectories"
    print("ok  sampling determinism / guidance / sampler switch")


def test_sampling_against_bfloat16_weights():
    """Consumers freeze Poc2Mol in bfloat16 (Poc2MolInferenceBuilder._bind).

    The ODE state stays float32 regardless, so without an explicit cast the first
    convolution raises "Input type and weight type should be the same".
    """
    model = tiny_model().eval().to(torch.bfloat16)
    generator = torch.Generator().manual_seed(3)
    sample = model.sample(
        protein=torch.rand(2, 1, 8, 8, 8).to(torch.bfloat16),
        n_steps=3,
        generator=generator,
    )
    assert sample.dtype == torch.float32
    assert torch.isfinite(sample).all()
    print("ok  sampling works against bfloat16 weights")


def test_ligand_only_batch_builder():
    """ZINC pretraining must hand the model a zero pocket of the right width."""
    from src.data.common.voxelization.config import Vox2SmilesDataConfig
    from src.data.poc2mol.ligand_data_module import (
        LigandVoxelBatchBuilder,
        collate_ligand_records,
    )

    config = Vox2SmilesDataConfig(
        vox_size=2.0, box_dims=[8.0, 8.0, 8.0], has_protein=False,
        ligand_channel_names=["a", "b"], dtype=torch.float32,
    )
    records = [
        {
            "coords": torch.zeros(3, 2),
            "vdw_radii": torch.full((2,), 1.7),
            "channels": torch.tensor([[True, False], [False, True]]),
            "center": torch.zeros(3),
            "smiles_str": "CC",
        }
        for _ in range(2)
    ]
    batch = collate_ligand_records(records)
    out = LigandVoxelBatchBuilder(config, n_protein_channels=4)(batch)

    assert out["ligand"].shape == (2, 2, 4, 4, 4), out["ligand"].shape
    assert out["protein"].shape == (2, 4, 4, 4, 4), out["protein"].shape
    assert out["protein"].abs().sum() == 0, "pocket slot must be identically zero"
    assert out["ligand"].sum() > 0, "voxeliser produced an empty ligand grid"
    print("ok  ligand-only batch builder pads a zero pocket")


def test_divergence_alarm_fires():
    """The diagnostics must SEPARATE a diverging sampler from an untrained one.

    Sample Dice cannot: both score ~0. The trajectory RMS can, because every point of the
    probability path has RMS ~1 in model space whatever the model has learned.
    """
    model = tiny_model().eval()
    protein = torch.rand(2, 1, 8, 8, 8)
    generator = torch.Generator().manual_seed(11)
    x0 = torch.randn(2, 2, 8, 8, 8, generator=generator)

    # (a) untrained: the head is zero-initialised, so the ODE is the identity map. The
    #     samples are useless, but the trajectory is perfectly well behaved.
    _, healthy = model.sample(protein=protein, x0=x0, n_steps=8, return_stats=True)
    assert 0.5 < healthy["traj_rms_max"] < 2.0, healthy
    assert healthy["out_of_range"] < 1.0

    # (b) a runaway velocity field, which is what the real failure looks like.
    with torch.no_grad():
        model.model.final_conv.weight.normal_(std=50.0)
    _, diverged = model.sample(protein=protein, x0=x0, n_steps=8, return_stats=True)

    assert diverged["traj_rms_max"] > 10 * healthy["traj_rms_max"], \
        f"divergence alarm did not fire: {diverged}"
    assert diverged["out_of_range"] > 0.5, diverged
    assert diverged["max_abs_occ"] > 10, diverged

    # The clamp is the remedy, and it must bound the state without hiding the symptom:
    # out_of_range is measured on the clamped state here, but max_abs_occ stays bounded.
    model.sample_clamp = 2.0
    _, clamped = model.sample(protein=protein, x0=x0, n_steps=8, return_stats=True)
    assert clamped["max_abs_occ"] <= 2.0 * model.occupancy_scale + model.occupancy_shift + 1e-4, clamped
    assert clamped["traj_rms_max"] < diverged["traj_rms_max"], clamped
    print(f"ok  divergence alarm (healthy rms {healthy['traj_rms_max']:.2f}, "
          f"diverged {diverged['traj_rms_max']:.1f}, clamped {clamped['traj_rms_max']:.2f})")


def test_restoration_probe_beats_full_sampling():
    """Starting halfway along the TRUE path must be easier than starting from noise.

    That gap is the whole diagnostic: if `restore` is high while `sample` is low, the
    velocity field is fine and the trajectory is the problem.
    """
    torch.manual_seed(0)
    model = tiny_model(ema_decay=0.0)
    ligand = torch.zeros(1, 2, 8, 8, 8)
    ligand[0, 0, 2:5, 2:5, 2:5] = 1.0
    protein = torch.zeros(1, 1, 8, 8, 8)
    protein[0, 0, 0:3, 0:3, 0:3] = 1.0

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(60):  # deliberately UNDER-trained
        optimizer.zero_grad()
        model.flow_loss(ligand, protein, drop_condition=False)["loss"].backward()
        optimizer.step()

    model.eval()
    x0 = torch.randn(1, 2, 8, 8, 8, generator=torch.Generator().manual_seed(5))
    from_noise = pooled_soft_dice(
        model.sample(protein=protein, x0=x0, n_steps=16), ligand
    ).item()
    x_half = 0.5 * x0 + 0.5 * model.to_model_space(ligand)
    from_half = pooled_soft_dice(
        model.sample(protein=protein, x0=x_half, t_start=0.5, n_steps=16), ligand
    ).item()
    assert from_half > from_noise, f"restore {from_half:.3f} !> sample {from_noise:.3f}"
    print(f"ok  restoration probe (from t=0.5: {from_half:.3f} vs from noise: {from_noise:.3f})")


def test_ema_contract():
    model = tiny_model(ema_decay=0.9, ema_warmup_steps=0)
    before = {k: v.clone() for k, v in model.state_dict().items()}

    model._ema_update()                       # seeds the shadow from the init weights
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p))       # "train"
    model._ema_update()

    name = next(iter(model._ema_shadow))
    assert not torch.allclose(model._ema_shadow[name], dict(model.named_parameters())[name]), \
        "EMA tracked the online weights exactly"

    # Validation runs on the EMA weights and training resumes on the online ones.
    online = {k: v.clone() for k, v in model.state_dict().items()}
    model.on_validation_epoch_start()
    swapped = dict(model.named_parameters())[name].clone()
    assert torch.allclose(swapped, model._ema_shadow[name].to(swapped.dtype))
    model.on_validation_epoch_end()
    assert torch.allclose(dict(model.named_parameters())[name], online[name])

    # A weights-only checkpoint carries the EMA weights, which is what validation measured.
    checkpoint = {"state_dict": {k: v.clone() for k, v in model.state_dict().items()}}
    model.on_save_checkpoint(checkpoint)
    assert torch.allclose(
        checkpoint["state_dict"][name], model._ema_shadow[name].to(before[name].dtype)
    )
    assert "raw_state_dict" not in checkpoint, "weights-only checkpoint was doubled in size"

    # A full checkpoint keeps the online weights too, so a resume is exact.
    full = {"state_dict": {k: v.clone() for k, v in model.state_dict().items()},
            "optimizer_states": []}
    model.on_save_checkpoint(full)
    assert torch.allclose(full["raw_state_dict"][name], online[name])
    print("ok  EMA update / swap / checkpoint contract")


def test_learns_a_conditional_density():
    """Train a tiny model on one (pocket, ligand) pair and check the samples converge.

    Deliberately a memorisation task: with a single example the conditional distribution is
    a point mass, so a correct flow implementation must be able to reproduce it almost
    exactly. A sign error in the path, the target velocity or the ODE direction shows up
    here as a Dice that does not move.
    """
    torch.manual_seed(0)
    model = tiny_model(protein_channels=1, ema_decay=0.0)

    ligand = torch.zeros(1, 2, 8, 8, 8)
    ligand[0, 0, 2:5, 2:5, 2:5] = 1.0
    ligand[0, 1, 5:7, 1:3, 4:6] = 1.0
    protein = torch.zeros(1, 1, 8, 8, 8)
    protein[0, 0, 0:3, 0:3, 0:3] = 1.0

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(400):
        optimizer.zero_grad()
        loss = model.flow_loss(ligand, protein, drop_condition=False)["loss"]
        loss.backward()
        optimizer.step()

    model.eval()
    generator = torch.Generator().manual_seed(7)
    sample = model.sample(protein=protein, n_steps=32, generator=generator)
    dice = pooled_soft_dice(sample, ligand).item()
    assert dice > 0.6, f"flow failed to memorise a single example: dice={dice:.3f}"
    print(f"ok  learns a conditional density (dice={dice:.3f}, final loss={loss.item():.4f})")


if __name__ == "__main__":
    test_unet_shapes_and_zero_init()
    test_unconditional_unet_rejects_condition()
    test_time_shift()
    test_occupancy_roundtrip()
    test_flow_loss_and_gradients()
    test_sampling_determinism_and_guidance()
    test_sampling_against_bfloat16_weights()
    test_ligand_only_batch_builder()
    test_divergence_alarm_fires()
    test_restoration_probe_beats_full_sampling()
    test_ema_contract()
    test_learns_a_conditional_density()
    print("\nall flow-matching checks passed")
