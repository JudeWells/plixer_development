import contextlib

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem, DataStructs


def _blocked_rdkit_logs():
    """Scoped suppression of RDKit's parse-error chatter.

    ``rdBase.BlockLogs`` is a context manager in recent RDKit but was a plain object in
    older builds, so fall back to a no-op rather than breaking on a version difference.
    """
    blocker = getattr(rdBase, "BlockLogs", None)
    if blocker is None:
        return contextlib.nullcontext()
    try:
        instance = blocker()
    except Exception:
        return contextlib.nullcontext()
    if hasattr(instance, "__enter__"):
        return instance
    # Older RDKit: the block lasts while the object is alive, so keep a reference for
    # the duration of the with-block and drop it on exit.
    return contextlib.nullcontext(instance)


def calculate_exact_match(generated_smiles, reference_smiles):
    """Fraction of generations that are the SAME MOLECULE as their reference.

    This is the honest readout of stage-1 progress. Token accuracy saturates (>0.99) and
    validity saturates (>0.999) long before the model is actually reconstructing molecules,
    so neither discriminates late in training; whole-molecule exact match still does.

    Compared after RDKit canonicalisation rather than as raw strings, because one molecule
    has many valid SMILES and string comparison scores the equivalent ones as misses. On
    ZINC the two agree to ~0.25 pp (the model reproduces canonical form, having been trained
    on it), but canonicalisation is the defensible definition.

    A generation that does not parse counts as a miss, never as an error -- early in
    training most of them will not parse.

    Args:
        generated_smiles: model outputs, positionally aligned with ``reference_smiles``.
            ``generate_smiles`` returns a list of the same length as its input batch, so
            the alignment holds; ``None`` entries are treated as misses.
        reference_smiles: ground-truth SMILES.

    Returns:
        Fraction in [0, 1], or nan if there is nothing to score.
    """
    n = min(len(generated_smiles), len(reference_smiles))
    if n == 0:
        return float("nan")

    # Early in training essentially every generation is malformed, and RDKit logs a
    # multi-line parse error for each one. BlockLogs scopes the suppression to this call
    # rather than muting RDKit globally for the process.
    with _blocked_rdkit_logs():
        matches = 0
        for gen, ref in zip(generated_smiles[:n], reference_smiles[:n]):
            if not gen or not ref:
                continue
            gen_mol = Chem.MolFromSmiles(gen)
            if gen_mol is None:
                continue  # unparseable generation is a miss
            ref_mol = Chem.MolFromSmiles(ref)
            if ref_mol is None:
                continue
            if Chem.MolToSmiles(gen_mol) == Chem.MolToSmiles(ref_mol):
                matches += 1
    return matches / n


def calculate_validity(smiles_list):
    """
    Calculate the percentage of valid SMILES strings.
    
    Args:
        smiles_list: List of SMILES strings
        
    Returns:
        Percentage of valid SMILES strings
    """
    if not smiles_list:
        return 0.0

    valid_count = 0
    # Same reason as calculate_exact_match: an untrained decoder produces a multi-line
    # RDKit parse error per sample, which drowns the training log.
    with _blocked_rdkit_logs():
        for smiles in smiles_list:
            if smiles and len(smiles) > 0:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    valid_count += 1
    
    return valid_count / len(smiles_list)


def calculate_uniqueness(smiles_list):
    """
    Calculate the percentage of unique SMILES strings.
    
    Args:
        smiles_list: List of SMILES strings
        
    Returns:
        Percentage of unique SMILES strings
    """
    if not smiles_list:
        return 0.0
    
    # Filter out invalid SMILES
    valid_smiles = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            # Canonicalize SMILES
            canonical_smiles = Chem.MolToSmiles(mol)
            valid_smiles.append(canonical_smiles)
    
    if not valid_smiles:
        return 0.0
    
    # Count unique SMILES
    unique_smiles = set(valid_smiles)
    
    return len(unique_smiles) / len(valid_smiles)


def calculate_novelty(generated_smiles, reference_smiles):
    """
    Calculate the percentage of generated SMILES that are not in the reference set.
    
    Args:
        generated_smiles: List of generated SMILES strings
        reference_smiles: List of reference SMILES strings
        
    Returns:
        Percentage of novel SMILES strings
    """
    if not generated_smiles:
        return 0.0
    
    # Filter out invalid SMILES and canonicalize
    valid_generated = []
    for smiles in generated_smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            canonical_smiles = Chem.MolToSmiles(mol)
            valid_generated.append(canonical_smiles)
    
    if not valid_generated:
        return 0.0
    
    # Canonicalize reference SMILES
    reference_set = set()
    for smiles in reference_smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            canonical_smiles = Chem.MolToSmiles(mol)
            reference_set.add(canonical_smiles)
    
    # Count novel SMILES
    novel_count = 0
    for smiles in valid_generated:
        if smiles not in reference_set:
            novel_count += 1
    
    return novel_count / len(valid_generated)


def calculate_similarity(smiles1, smiles2):
    """
    Calculate the Tanimoto similarity between two SMILES strings.
    
    Args:
        smiles1: First SMILES string
        smiles2: Second SMILES string
        
    Returns:
        Tanimoto similarity (0-1)
    """
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    
    if mol1 is None or mol2 is None:
        return 0.0
    
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
    
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def calculate_paired_similarity(generated_smiles, reference_smiles):
    """Mean Morgan-Tanimoto between each generation and ITS OWN reference.

    Deliberately NOT `calculate_average_similarity`, which takes the max over the entire
    reference set: that answers "does this molecule resemble anything in the batch", is
    inflated by batch size, and is blind to whether the model got the *right* molecule for
    the *right* pocket. For judging conditioned generation the pairing is the whole point.

    Invalid generations score 0 rather than being dropped. Excluding them would let a model
    that emits garbage half the time report a better similarity than one that always emits
    something reasonable.

    Fingerprints are Morgan radius 2, 2048 bits -- the same as the published evaluation, so
    numbers stay comparable with the paper's similarity-enrichment figures.
    """
    n = min(len(generated_smiles), len(reference_smiles))
    if n == 0:
        return float("nan")

    with _blocked_rdkit_logs():
        total = 0.0
        for gen, ref in zip(generated_smiles[:n], reference_smiles[:n]):
            if not gen or not ref:
                continue
            gen_mol = Chem.MolFromSmiles(gen)
            ref_mol = Chem.MolFromSmiles(ref)
            if gen_mol is None or ref_mol is None:
                continue  # counts as 0.0
            gen_fp = AllChem.GetMorganFingerprintAsBitVect(gen_mol, 2, nBits=2048)
            ref_fp = AllChem.GetMorganFingerprintAsBitVect(ref_mol, 2, nBits=2048)
            total += DataStructs.TanimotoSimilarity(gen_fp, ref_fp)
    return total / n


def calculate_average_similarity(generated_smiles, reference_smiles):
    """
    Calculate the average Tanimoto similarity between generated SMILES and reference SMILES.
    
    Args:
        generated_smiles: List of generated SMILES strings
        reference_smiles: List of reference SMILES strings
        
    Returns:
        Average Tanimoto similarity (0-1)
    """
    if not generated_smiles or not reference_smiles:
        return 0.0
    
    # Filter out invalid SMILES
    valid_generated = []
    for smiles in generated_smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            valid_generated.append(smiles)
    
    valid_reference = []
    for smiles in reference_smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            valid_reference.append(smiles)
    
    if not valid_generated or not valid_reference:
        return 0.0
    
    # Calculate similarities
    similarities = []
    for gen_smiles in valid_generated:
        max_sim = 0.0
        for ref_smiles in valid_reference:
            sim = calculate_similarity(gen_smiles, ref_smiles)
            max_sim = max(max_sim, sim)
        similarities.append(max_sim)
    
    return np.mean(similarities)


def calculate_metrics(generated_smiles, reference_smiles=None):
    """
    Calculate all metrics for generated SMILES.
    
    Args:
        generated_smiles: List of generated SMILES strings
        reference_smiles: List of reference SMILES strings (optional)
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        "validity": calculate_validity(generated_smiles),
        "uniqueness": calculate_uniqueness(generated_smiles),
    }
    
    if reference_smiles is not None:
        metrics["novelty"] = calculate_novelty(generated_smiles, reference_smiles)
        metrics["avg_similarity"] = calculate_average_similarity(generated_smiles, reference_smiles)
    
    return metrics


def accuracy_from_outputs(
    model_outputs,
    input_ids,
    start_ix=0,
    ignore_index=-100,
    dataset_names=None,
):
    """Compute the accuracy of the target sequence given the model outputs.
    Args:
        model_outputs: The model outputs from the forward pass.
        input_ids: The input sequence.
        ignore_index: Token index to ignore when computing accuracy.
            (this will get added automatically by the data collator as padding)
    Returns:
        The accuracy of the target sequence.
    """
    logits = model_outputs.logits.detach()
    # Shift so that tokens < n predict n
    shift_logits = logits[..., start_ix:, :].contiguous()  # b, L, V
    shift_labels = input_ids[..., start_ix:].contiguous()  # b, L
    # Ensure tensors are on the same device
    shift_labels = shift_labels.to(shift_logits.device)
    non_padding_mask = shift_labels != ignore_index
    
    accuracy = (shift_logits.argmax(-1) == shift_labels).float()
    if dataset_names is not None:
        ds_accuracies = {}
        for ds_name in set(dataset_names):
            in_dataset_mask = np.array(dataset_names) == ds_name
            ds_accuracies[ds_name] = (
                accuracy[in_dataset_mask] * non_padding_mask[in_dataset_mask]
            ).sum() / non_padding_mask[in_dataset_mask].sum()
        return ds_accuracies
    accuracy = (accuracy * non_padding_mask).sum() / non_padding_mask.sum()
    return accuracy

