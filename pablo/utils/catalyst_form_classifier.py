"""SMILES-based classification of NHC pre-catalyst forms and free-carbene construction.

Four forms are recognised:
- ``salt``       : imidazolium/imidazolinium salt with an ionic counterion
                   (Cl⁻, Br⁻, BF₄⁻, PF₆⁻, etc.)
- ``free_nhc``   : free N-heterocyclic carbene (carbene carbon encoded as [C])
- ``co2_adduct`` : betaine/zwitterion from NHC + CO₂ (no explicit counterion)
- ``bicarbonate``: salt with HCO₃⁻ counterion

These heuristics cover the four forms present in the 161-molecule library.
Edge cases and classification results for the 22 tested molecules are printed
separately by ``classify_library`` for supervisor verification.

This module also provides ``to_free_carbene``, an RDKit-based routine that
deprotonates a salt/bicarbonate cation or strips the carboxylate of a
CO₂-adduct betaine to yield the neutral free-carbene SMILES. It walks the
molecular graph, so it does not depend on how the SMILES string is written.

IMPORTANT: Do NOT use these labels downstream without supervisor sign-off on the
22 tested catalysts.  The heuristics may misclassify unusual SMILES representations.
"""

from __future__ import annotations

import warnings

from rdkit import Chem


# SMARTS for the bicarbonate anion fragment (HCO3-).
_BICARBONATE_SMARTS = Chem.MolFromSmarts("[O-]C(=O)O")
# SMARTS for a carboxylate group (C(=O)[O-]) present in CO2-adduct betaines.
# We use just the carboxylate fragment rather than the full [N+]=C-carboxylate path
# because the N+=C bond traversal fails on ring-closure atoms (e.g. C6, C7, C72, C97).
_CARBOXYLATE_SMARTS = Chem.MolFromSmarts("C(=O)[O-]")


def classify_precatalyst_form(smiles: str) -> str:
    """Classify a pre-catalyst SMILES into one of four NHC forms.

    Priority order: salt/bicarbonate (multi-fragment) > co2_adduct (single fragment
    betaine) > free_nhc (literal [C] or [C-] in SMILES string) > unknown.

    CO2-adduct detection: single fragment + net charge 0 + carboxylate ``C(=O)[O-]``
    substructure. Only the carboxylate moiety is matched, because a full
    ``[#7+]=[#6]C(=O)[O-]`` SMARTS misses ring-closure bonds (e.g. C6, C7, C72, C97).

    Free-carbene detection uses SMILES TEXT checks (``'[C]' in smiles`` or
    ``'[C-]' in smiles``), not a SMARTS match.  Reason: ``Chem.MolFromSmarts('[C]')``
    matches ANY aliphatic carbon.  ``[C]`` is the neutral carbene notation;
    ``[C-]`` is used by some sources for the carbanion/free-carbene form (C50/C67/C80).

    Args:
        smiles: SMILES string of the pre-catalyst.

    Returns:
        One of ``'free_nhc'``, ``'co2_adduct'``, ``'bicarbonate'``,
        ``'salt'``, or ``'unknown'``.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "unknown"

    fragments = smiles.split(".")

    # ── Multi-fragment molecules ──────────────────────────────────────────────
    if len(fragments) > 1:
        # Bicarbonate: one of the fragments is HCO3-
        if mol.HasSubstructMatch(_BICARBONATE_SMARTS):
            return "bicarbonate"
        # Salt: any fragment is a net-anion
        for frag_smi in fragments:
            frag_mol = Chem.MolFromSmiles(frag_smi)
            if frag_mol is not None:
                net_charge = sum(a.GetFormalCharge() for a in frag_mol.GetAtoms())
                if net_charge < 0:
                    return "salt"
        return "unknown"

    # ── Single-fragment molecules ─────────────────────────────────────────────
    # CO2 adduct: net-neutral betaine with a carboxylate group.
    # We test carboxylate presence rather than the full [N+]=C path because the
    # latter fails when the N+=C bond straddles a ring-closure in SMILES notation.
    net_charge = sum(a.GetFormalCharge() for a in mol.GetAtoms())
    if net_charge == 0 and mol.HasSubstructMatch(_CARBOXYLATE_SMARTS):
        return "co2_adduct"

    # Inner salt (e.g. sydnone, mesoionic compound): single fragment, net charge 0,
    # but carries explicit positive and negative formal charges — not a CO2 adduct.
    has_pos = any(a.GetFormalCharge() > 0 for a in mol.GetAtoms())
    has_neg = any(a.GetFormalCharge() < 0 for a in mol.GetAtoms())
    if has_pos and has_neg:
        return "salt"

    # Free NHC: carbene carbon written as [C] (neutral) or [C-] (carbanion form).
    # Text checks are intentional — SMARTS '[C]' matches ALL aliphatic carbons.
    if "[C]" in smiles or "[C-]" in smiles:
        return "free_nhc"

    return "unknown"


# One-hot encoding order — must remain fixed across all runs.
FORM_LABELS = ("salt", "free_nhc", "co2_adduct", "bicarbonate")


def form_to_onehot(form: str) -> list[int]:
    """Convert a form label to a 4-dim one-hot vector.

    Order: [salt, free_nhc, co2_adduct, bicarbonate].
    Unknown forms are encoded as all-zeros.

    Args:
        form: One of the four form labels or ``'unknown'``.

    Returns:
        4-element list of 0/1 integers.
    """
    vec = [0, 0, 0, 0]
    if form in FORM_LABELS:
        vec[FORM_LABELS.index(form)] = 1
    return vec


def form_to_index(form: str) -> int:
    """Convert a form label to a zero-indexed integer category.

    Order: salt=0, free_nhc=1, co2_adduct=2, bicarbonate=3.
    Unknown forms default to 0.

    Args:
        form: One of the four form labels or ``'unknown'``.

    Returns:
        Integer in [0, 3].
    """
    if form in FORM_LABELS:
        return FORM_LABELS.index(form)
    return 0


def classify_library(
    catalyst_ids: list[int],
    smiles_list: list[str],
    tested_ids: set[int] | None = None,
) -> dict[int, str]:
    """Classify all library molecules and print a verification table for tested IDs.

    Args:
        catalyst_ids: List of integer catalyst IDs (matches xlsx ID column).
        smiles_list: Corresponding list of SMILES strings.
        tested_ids: Set of catalyst IDs tested in the experimental dataset.
                    Their classifications are printed for supervisor sign-off.

    Returns:
        Dict mapping catalyst_id → form label.
    """
    classifications: dict[int, str] = {}

    for cid, smi in zip(catalyst_ids, smiles_list):
        classifications[cid] = classify_precatalyst_form(smi)

    # Tally per form
    from collections import Counter
    counts = Counter(classifications.values())
    print("Pre-catalyst form counts across 161-molecule library:")
    for form in (*FORM_LABELS, "unknown"):
        print(f"  {form:<15}: {counts[form]}")

    # Print verification table for tested catalysts
    if tested_ids:
        print("\nForm classifications for the 22 TESTED catalysts — verify with supervisor:")
        print(f"  {'ID':<6} {'Form':<15} {'SMILES[:60]'}")
        print(f"  {'-'*6} {'-'*15} {'-'*60}")
        for cid, smi in zip(catalyst_ids, smiles_list):
            if cid in tested_ids:
                form = classifications[cid]
                flag = "  ← VERIFY" if form == "unknown" else ""
                print(f"  C{cid:<5} {form:<15} {smi[:60]}{flag}")

    return classifications


# ── NHC ring location and free-carbene construction ──────────────────────────


_NHC_HETEROATOMS = (7, 8, 16)  # N, O, S — covers imidazol(in)ium, oxazolium, thiazolium, triazolium


def locate_carbene_center(
    mol: Chem.Mol,
) -> tuple[int | None, list[int], list[int], set[int]]:
    """Locate the carbene center and its neighbors for both cyclic and acyclic NHCs.

    The carbene C is identified as a carbon bonded to ≥2 heteroatoms (N, O, S).
    Priority is given to 5- or 6-membered rings. If no ring is found, acyclic
    centers are identified.

    Args:
        mol: RDKit ``Mol`` with up-to-date ring info.

    Returns:
        Tuple ``(carbene_c_idx, hetero_neighbor_idxs, backbone_idxs, ring_atom_set)``.
        For acyclic NHCs, ``backbone_idxs`` is empty and ``ring_atom_set`` contains
        only the carbene C.
    """
    # 1. Try to find a 5- or 6-membered ring carbene (imidazolylidene etc.)
    ring_info = mol.GetRingInfo()
    for ring in ring_info.AtomRings():
        if len(ring) not in (5, 6):
            continue
        atoms = [mol.GetAtomWithIdx(i) for i in ring]
        if not any(a.GetAtomicNum() == 7 for a in atoms):
            continue
        for i in ring:
            a = mol.GetAtomWithIdx(i)
            if a.GetAtomicNum() != 6:
                continue
            hetero_neighbors = [
                n.GetIdx()
                for n in a.GetNeighbors()
                if n.GetIdx() in ring and n.GetAtomicNum() in _NHC_HETEROATOMS
            ]
            if len(hetero_neighbors) >= 1 and a.GetDegree() == 2:
                # Divalent carbon with at least one ring heteroatom:
                # Matches CAACs (1 N) and NHCs (2 N/O/S).
                ring_atoms = set(ring)
                backbone = [j for j in ring if j != i and j not in hetero_neighbors]
                return i, hetero_neighbors, backbone, ring_atoms

    # 2. Acyclic fallback: search for a divalent neutral carbon with 2 hetero-neighbors
    for atom in mol.GetAtoms():
        if (
            atom.GetAtomicNum() == 6
            and atom.GetFormalCharge() == 0
            and atom.GetNoImplicit()
        ):
            hetero_neighbors = [
                n.GetIdx()
                for n in atom.GetNeighbors()
                if n.GetAtomicNum() in _NHC_HETEROATOMS
            ]
            if len(hetero_neighbors) >= 2:
                # For acyclic, ring_atom_set contains only the carbene C to satisfy
                # the "exclude from substituent" logic in Sterimol.
                return atom.GetIdx(), hetero_neighbors, [], {atom.GetIdx()}

    return None, [], [], set()


def _largest_fragment(mol: Chem.Mol) -> Chem.Mol:
    """Return the largest fragment of ``mol`` by heavy-atom count.

    Ring info is re-perceived on the returned fragment so callers can rely on
    ``GetRingInfo()``.

    Args:
        mol: RDKit ``Mol``, possibly multi-fragment (e.g. salt + counterion).

    Returns:
        The largest fragment as a fresh ``Mol`` with up-to-date ring info.
    """
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    largest = max(frags, key=lambda m: m.GetNumHeavyAtoms())
    largest.UpdatePropertyCache(strict=False)
    Chem.GetSSSR(largest)
    return largest


def _strip_co2_adduct(mol: Chem.Mol) -> Chem.Mol:
    """Remove the carboxylate (``C(=O)[O-]``) group from a CO₂-adduct betaine.

    The carboxylate C is attached to the carbene C in a betaine.  We delete the
    carboxylate carbon and its two oxygens; the carbene C is left in the same
    state as in the corresponding salt cation, ready for the salt-style
    deprotonation routine.

    Args:
        mol: RDKit ``Mol`` of a single-fragment CO₂-adduct.

    Returns:
        Editable ``Mol`` with the C(=O)[O-] group removed.
    """
    matches = mol.GetSubstructMatches(_CARBOXYLATE_SMARTS)
    assert matches, "No carboxylate group found in claimed CO₂ adduct."
    # First match: (carboxylate_C, =O, [O-])  — see SMARTS atom order.
    carboxylate_atoms = list(matches[0])
    rw = Chem.RWMol(mol)
    # Delete in descending index order so earlier indices remain valid.
    for idx in sorted(carboxylate_atoms, reverse=True):
        rw.RemoveAtom(idx)
    out = rw.GetMol()
    out.UpdatePropertyCache(strict=False)
    Chem.GetSSSR(out)
    return out


def _locate_iminium_carbon(
    mol: Chem.Mol,
    require_charged: bool = False,
) -> tuple[int | None, list[int], set[int]]:
    """Locate the carbon double-bonded to a nitrogen (the iminium C).

    This is the carbene-C candidate in any NHC salt cation, including CAACs
    (one ring N), thiazoliums, imidazoliums, triazoliums, and acyclic
    formamidiniums (no ring).  The unifying signature is the iminium C=N+ bond.
    Ring matches are preferred over acyclic; charged-N matches are preferred
    over neutral-N.

    Args:
        mol: RDKit ``Mol`` (Kekulized) representing the cation.
        require_charged: If True, accept only C=N⁺ candidates (i.e. the partner
            N has formal charge ≥ 1).  Used during iterative deprotonation of
            bis-NHCs to avoid further deprotonating residual conjugated C=N
            bonds in 1,2,4-triazolylidenes etc.

    Returns:
        Tuple ``(carbene_c_idx, neighbour_nitrogen_idxs, ring_atom_set)``.
        ``ring_atom_set`` is empty for acyclic matches.  If no candidate is
        found, ``(None, [], set())``.
    """
    ring_info = mol.GetRingInfo()
    best_ring_charged: tuple[int, list[int], set[int]] | None = None
    best_ring_neutral: tuple[int, list[int], set[int]] | None = None
    for ring in ring_info.AtomRings():
        if len(ring) not in (5, 6):
            continue
        ring_set = set(ring)
        ring_ns = [i for i in ring if mol.GetAtomWithIdx(i).GetAtomicNum() == 7]
        if not ring_ns:
            continue
        for i in ring:
            if mol.GetAtomWithIdx(i).GetAtomicNum() != 6:
                continue
            for n_idx in ring_ns:
                bond = mol.GetBondBetweenAtoms(i, n_idx)
                if bond is None or bond.GetBondType() != Chem.BondType.DOUBLE:
                    continue
                cand = (i, ring_ns, ring_set)
                if mol.GetAtomWithIdx(n_idx).GetFormalCharge() > 0:
                    if best_ring_charged is None:
                        best_ring_charged = cand
                else:
                    if best_ring_neutral is None:
                        best_ring_neutral = cand

    if best_ring_charged is not None:
        return best_ring_charged
    if not require_charged and best_ring_neutral is not None:
        return best_ring_neutral

    # Acyclic fallback: any C=N+ bond, regardless of ring membership.
    best_acyc_charged: tuple[int, list[int], set[int]] | None = None
    best_acyc_neutral: tuple[int, list[int], set[int]] | None = None
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.DOUBLE:
            continue
        a1, a2 = bond.GetBeginAtom(), bond.GetEndAtom()
        if a1.GetAtomicNum() == 6 and a2.GetAtomicNum() == 7:
            c_atom, n_atom = a1, a2
        elif a2.GetAtomicNum() == 6 and a1.GetAtomicNum() == 7:
            c_atom, n_atom = a2, a1
        else:
            continue
        c_idx = c_atom.GetIdx()
        all_n_neighbours = [
            nbr.GetIdx() for nbr in c_atom.GetNeighbors() if nbr.GetAtomicNum() == 7
        ]
        cand = (c_idx, all_n_neighbours, set())
        if n_atom.GetFormalCharge() > 0:
            if best_acyc_charged is None:
                best_acyc_charged = cand
        else:
            if best_acyc_neutral is None:
                best_acyc_neutral = cand

    if best_acyc_charged is not None:
        return best_acyc_charged
    if not require_charged and best_acyc_neutral is not None:
        return best_acyc_neutral
    return (None, [], set())


def _deprotonate_one_iminium(
    rw: Chem.RWMol, processed: set[int], require_charged: bool = True
) -> bool:
    """Deprotonate one iminium centre in ``rw`` (in-place).  Returns True if a
    centre was processed, False if none remained.

    Bis-NHC pre-catalysts contain two iminium carbons; this helper is meant to
    be called in a loop until it returns False.  When called iteratively, set
    ``require_charged=True`` so residual conjugated C=N bonds (e.g. in the
    1,2,4-triazolylidene scaffold) are not picked up as additional carbene
    candidates after the genuine iminium has been deprotonated.

    Args:
        rw: RWMol to modify in place.
        processed: Set of carbene-C atom indices already deprotonated; skip these.
        require_charged: If True, only deprotonate centres where the partner N
            still carries formal charge ≥ 1.

    Returns:
        True if a new iminium centre was found and deprotonated, False otherwise.
    """
    Chem.GetSSSR(rw)
    carbene_idx, ring_ns, _ = _locate_iminium_carbon(rw, require_charged=require_charged)
    if carbene_idx is None or carbene_idx in processed:
        return False

    cC = rw.GetAtomWithIdx(carbene_idx)
    cC.SetFormalCharge(0)
    cC.SetNumExplicitHs(0)
    cC.SetNoImplicit(True)

    for n_idx in ring_ns:
        n_atom = rw.GetAtomWithIdx(n_idx)
        if n_atom.GetFormalCharge() > 0:
            n_atom.SetFormalCharge(0)

    for n_idx in ring_ns:
        bond = rw.GetBondBetweenAtoms(carbene_idx, n_idx)
        if bond is not None and bond.GetBondType() == Chem.BondType.DOUBLE:
            bond.SetBondType(Chem.BondType.SINGLE)

    processed.add(carbene_idx)
    return True


def _deprotonate_to_carbene(mol: Chem.Mol) -> tuple[Chem.Mol, int]:
    """Convert an NHC cation/zwitterion to its free-carbene form.

    Iterates over all iminium centres, so bis-NHC pre-catalysts (two linked
    NHC rings) get both centres deprotonated.

    Args:
        mol: RDKit ``Mol`` representing the cation (or post-CO₂-strip zwitterion).

    Returns:
        ``(neutral_carbene_mol, net_charge)``.  ``net_charge`` will be 0 for
        typical NHCs, +1/+2 if bis-NHC deprotonation is incomplete (caller
        should fall back), or −1 for mesoionic / [N⁻] structures.
    """
    Chem.Kekulize(mol, clearAromaticFlags=True)
    rw = Chem.RWMol(mol)
    processed: set[int] = set()

    deprotonated_any = False
    # First pass: only act on charged iminium (C=N+) centres.  This handles
    # bis-NHCs cleanly while leaving conjugated neutral C=N bonds intact (e.g.
    # 1,2,4-triazolylidenes where N=C is part of the aromatic system, not a
    # second iminium).
    for _ in range(8):
        if not _deprotonate_one_iminium(rw, processed, require_charged=True):
            break
        deprotonated_any = True
    # Second pass: if no charged iminium was ever found (e.g. unusual SMILES
    # with the charge implicit), accept a neutral C=N as the carbene candidate.
    if not deprotonated_any:
        if _deprotonate_one_iminium(rw, processed, require_charged=False):
            deprotonated_any = True

    if not deprotonated_any:
        # Fallback: pre-existing [C] free-carbene atoms; nothing to do beyond cleanup.
        Chem.GetSSSR(rw)
        carbene_idx, ring_ns, _, _ = locate_carbene_center(rw)
        assert carbene_idx is not None, "No iminium / NHC ring found while deprotonating."
        cC = rw.GetAtomWithIdx(carbene_idx)
        cC.SetFormalCharge(0)
        cC.SetNumExplicitHs(0)
        cC.SetNoImplicit(True)

    out_mol = rw.GetMol()
    out_mol.UpdatePropertyCache(strict=False)
    Chem.SanitizeMol(
        out_mol,
        sanitizeOps=Chem.SanitizeFlags.SANITIZE_FINDRADICALS
        | Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION
        | Chem.SanitizeFlags.SANITIZE_SETCONJUGATION
        | Chem.SanitizeFlags.SANITIZE_SYMMRINGS,
    )
    net_charge = sum(a.GetFormalCharge() for a in out_mol.GetAtoms())
    return out_mol, net_charge


def to_free_carbene(smiles: str, form: str) -> tuple[str, int]:
    """Convert a pre-catalyst SMILES to the corresponding free-carbene SMILES.

    Walks the molecular graph via RDKit rather than matching SMILES substrings,
    so 1,3-disubstituted imidazoliums, 1,2,4-triazoliums, thiazoliums, and
    bicarbonates all transform correctly. Every iminium centre is deprotonated,
    so bis-NHC pre-catalysts give the neutral bis-carbene.

    Args:
        smiles: Pre-catalyst SMILES from the library.
        form: One of ``'salt'``, ``'free_nhc'``, ``'co2_adduct'``, ``'bicarbonate'``,
            ``'unknown'``.

    Returns:
        ``(free_carbene_smiles, net_charge)``.  ``net_charge`` is the formal
        charge of the returned species (normally 0, but −1 for SMILES that use
        ``[C-]`` carbanion notation for the carbene).  Pass ``net_charge`` into
        downstream xTB/DFT calls as ``--chrg``.

        On any validation failure, returns the canonical SMILES of the input's
        largest fragment with charge 0 and emits a warning.
    """
    if form not in ("salt", "free_nhc", "co2_adduct", "bicarbonate", "unknown"):
        raise ValueError(f"Unknown form label: {form!r}")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        warnings.warn(f"to_free_carbene: RDKit failed to parse input SMILES: {smiles}")
        return smiles, 0

    if form == "free_nhc":
        cation = _largest_fragment(mol)
        canon = Chem.MolToSmiles(cation)
        net = sum(a.GetFormalCharge() for a in cation.GetAtoms())
        return canon, net

    if form == "unknown":
        warnings.warn(
            f"to_free_carbene: form='unknown' for SMILES {smiles!r} — returning largest fragment unchanged."
        )
        cation = _largest_fragment(mol)
        return Chem.MolToSmiles(cation), 0

    if form in ("salt", "bicarbonate"):
        cation = _largest_fragment(mol)
    else:  # form == "co2_adduct"
        cation = _strip_co2_adduct(mol)

    # [VERIFY]: SanitizeMol with SANITIZE_PROPERTIES disabled is necessary because the
    # divalent carbene C has a non-standard valence that the default sanitizer rejects.
    free, net = _deprotonate_to_carbene(cation)
    canon = Chem.MolToSmiles(free)

    # Validation: round-trip must parse, and the result must contain a divalent
    # carbon bonded to ≥1 heteroatom (N/O/S) — the free-carbene signature.  A
    # strict NHC-ring check would reject CAACs (1 ring N) and acyclic
    # diaminocarbenes, so we use the broader divalent-C criterion.
    check = Chem.MolFromSmiles(canon, sanitize=False)
    if check is None:
        warnings.warn(
            f"to_free_carbene: result unparseable for {smiles!r} (form={form}); "
            "falling back to original SMILES."
        )
        return Chem.MolToSmiles(_largest_fragment(mol)), 0
    check.UpdatePropertyCache(strict=False)
    has_carbene = any(
        a.GetAtomicNum() == 6
        and a.GetTotalNumHs() == 0
        and a.GetFormalCharge() == 0
        and a.GetNoImplicit()
        and any(n.GetAtomicNum() in _NHC_HETEROATOMS for n in a.GetNeighbors())
        for a in check.GetAtoms()
    )
    if not has_carbene:
        warnings.warn(
            f"to_free_carbene: no divalent carbene C in output for {smiles!r} "
            f"(form={form}); falling back to original SMILES."
        )
        return Chem.MolToSmiles(_largest_fragment(mol)), 0

    return canon, net


def _locate_free_carbene_carbons(mol: Chem.Mol) -> list[tuple[int, list[int]]]:
    """Find every divalent, hydrogen-free carbene carbon and its heteroatom neighbours.

    Stricter than ``locate_carbene_center``, which accepts any divalent ring
    carbon with a heteroatom neighbour and therefore also matches backbone C–H
    carbons of the imidazole ring (both have degree 2). Requiring zero hydrogens
    isolates the genuine carbene centre. Returns every match so bis-NHCs are
    visible to the caller.

    Args:
        mol: RDKit ``Mol`` with up-to-date ring info.

    Returns:
        List of ``(carbene_c_idx, hetero_neighbor_idxs)``, ordered by atom index.
    """
    found: list[tuple[int, list[int]]] = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 6 or atom.GetDegree() != 2:
            continue
        if atom.GetTotalNumHs() != 0:
            continue
        hetero = [
            n.GetIdx()
            for n in atom.GetNeighbors()
            if n.GetAtomicNum() in _NHC_HETEROATOMS
        ]
        if hetero:
            found.append((atom.GetIdx(), hetero))
    return found


def locate_c2_center(mol: Chem.Mol) -> int | None:
    """Locate the C2 carbon in either the azolium or the free-carbene form.

    Dispatches on which species is present: a free carbene is found as a
    divalent hydrogen-free carbon, an azolium as its iminium C=N⁺ carbon. Needed
    because neither single criterion covers both — the protonated C2 is trivalent
    (so ``locate_carbene_center`` misses it), while CAACs carry only one ring
    nitrogen (so a "two ring heteroatoms" rule misses them).

    Call this before ``Chem.AddHs``: RDKit appends explicit hydrogens after the
    heavy atoms, so a heavy-atom index stays valid afterwards.

    Args:
        mol: RDKit ``Mol`` of either species, with up-to-date ring info.

    Returns:
        0-based index of the C2 carbon, or ``None`` if no candidate is found.
    """
    work = Chem.RWMol(mol)
    Chem.Kekulize(work, clearAromaticFlags=True)
    Chem.GetSSSR(work)

    carbenes = _locate_free_carbene_carbons(work)
    if carbenes:
        return carbenes[0][0]

    idx, _, _ = _locate_iminium_carbon(work, require_charged=True)
    if idx is None:
        idx, _, _ = _locate_iminium_carbon(work, require_charged=False)
    return idx


def _deprotonate_single_centre(mol: Chem.Mol) -> Chem.Mol:
    """Remove exactly one C2 proton from an azolium cation.

    Unlike ``_deprotonate_to_carbene``, which loops until every iminium centre is
    consumed, this stops after the first. For mono-NHCs the two are equivalent;
    for bis-NHC dications this yields the mono-carbene monocation rather than the
    neutral bis-carbene, keeping the deprotonation stoichiometry at one proton
    for every catalyst in the library.

    Args:
        mol: RDKit ``Mol`` of the azolium cation.

    Returns:
        RDKit ``Mol`` with a single C2 centre deprotonated.
    """
    Chem.Kekulize(mol, clearAromaticFlags=True)
    rw = Chem.RWMol(mol)
    processed: set[int] = set()

    if not _deprotonate_one_iminium(rw, processed, require_charged=True):
        assert _deprotonate_one_iminium(
            rw, processed, require_charged=False
        ), "No iminium centre found while deprotonating the azolium."

    out_mol = rw.GetMol()
    out_mol.UpdatePropertyCache(strict=False)
    Chem.SanitizeMol(
        out_mol,
        sanitizeOps=Chem.SanitizeFlags.SANITIZE_FINDRADICALS
        | Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION
        | Chem.SanitizeFlags.SANITIZE_SETCONJUGATION
        | Chem.SanitizeFlags.SANITIZE_SYMMRINGS,
    )
    return out_mol


def to_mono_carbene(smiles: str, form: str) -> tuple[str, int]:
    """Build the single-deprotonation partner of ``to_azolium``.

    Together these two define the ΔG_deprot thermodynamic cycle: the returned
    species differs from ``to_azolium(smiles, form)`` by exactly one C2 proton
    and exactly one unit of charge, for every form in the library. For
    mono-NHCs this reproduces ``to_free_carbene``; for the two bis-NHC
    dications it returns the mono-carbene monocation instead of the neutral
    bis-carbene, so the deprotonation stoichiometry stays uniform.

    Args:
        smiles: Pre-catalyst SMILES from the library.
        form: One of ``'salt'``, ``'free_nhc'``, ``'co2_adduct'``, ``'bicarbonate'``,
            ``'unknown'``.

    Returns:
        ``(carbene_smiles, net_charge)``, with ``net_charge`` exactly one less
        than the charge returned by ``to_azolium`` for the same input.
    """
    azolium_smi, _ = to_azolium(smiles, form)
    mol = Chem.MolFromSmiles(azolium_smi, sanitize=False)
    assert mol is not None, f"to_mono_carbene: unparseable azolium {azolium_smi!r}"
    mol.UpdatePropertyCache(strict=False)
    Chem.GetSSSR(mol)

    carbene = _deprotonate_single_centre(mol)
    return Chem.MolToSmiles(carbene), sum(
        a.GetFormalCharge() for a in carbene.GetAtoms()
    )


def _protonate_carbene(mol: Chem.Mol) -> Chem.Mol:
    """Protonate every free-carbene centre at C2 to give the azolium cation.

    Inverse of ``_deprotonate_one_iminium``. Two input notations occur in the
    library and are handled separately:

    * **Neutral carbene** (``[C]`` with two single C–N bonds): the carbon gains
      one explicit H, one ring C–N bond is promoted to a double bond, and that N
      takes the +1 charge. The charge goes on N rather than on C2 because
      ``[CH+]`` would be a carbocation, not an azolium.
    * **Ylide / mesoionic notation** (``[N+]=[C-]``): the C=N⁺ bond and the
      cation are already present, so the carbanion simply gains one H and is
      neutralised, taking the fragment from net 0 to net +1.

    Args:
        mol: RDKit ``Mol`` of the free carbene (neutral or ylide-notated).

    Returns:
        RDKit ``Mol`` of the azolium cation.
    """
    Chem.Kekulize(mol, clearAromaticFlags=True)
    rw = Chem.RWMol(mol)
    Chem.GetSSSR(rw)

    centres = _locate_free_carbene_carbons(rw)
    assert centres, "No free-carbene centre found while protonating."

    for carbene_idx, hetero_ns in centres:
        cC = rw.GetAtomWithIdx(carbene_idx)
        was_carbanion = cC.GetFormalCharge() < 0
        cC.SetNoImplicit(True)
        cC.SetNumExplicitHs(1)
        cC.SetFormalCharge(0)

        if was_carbanion:
            continue

        nitrogens = [i for i in hetero_ns if rw.GetAtomWithIdx(i).GetAtomicNum() == 7]
        partner = nitrogens[0] if nitrogens else hetero_ns[0]
        rw.GetAtomWithIdx(partner).SetFormalCharge(1)
        bond = rw.GetBondBetweenAtoms(carbene_idx, partner)
        assert bond is not None, "Carbene C is not bonded to its located heteroatom."
        bond.SetBondType(Chem.BondType.DOUBLE)

    out_mol = rw.GetMol()
    out_mol.UpdatePropertyCache(strict=False)
    Chem.SanitizeMol(
        out_mol,
        sanitizeOps=Chem.SanitizeFlags.SANITIZE_FINDRADICALS
        | Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION
        | Chem.SanitizeFlags.SANITIZE_SETCONJUGATION
        | Chem.SanitizeFlags.SANITIZE_SYMMRINGS,
    )
    return out_mol


def to_azolium(smiles: str, form: str) -> tuple[str, int]:
    """Convert a pre-catalyst SMILES to its azolium (C2-protonated) cation SMILES.

    The counterpart of ``to_free_carbene``, giving the other species of the
    C2-deprotonation equilibrium. For most of the library the azolium is already
    an intermediate of the free-carbene construction, so the same helpers are
    reused: ``salt`` / ``bicarbonate`` pre-catalysts *are* azolium cations once
    the counterion is dropped, and ``_strip_co2_adduct`` leaves the CO₂ adduct in
    the same state as the corresponding salt cation. Only ``free_nhc`` inputs
    need genuine protonation.

    Counterions, CO₂, and bicarbonate are deliberately excluded so that the
    azolium/carbene pair forms a clean thermodynamic cycle differing only in the
    C2 proton and the total charge.

    Args:
        smiles: Pre-catalyst SMILES from the library.
        form: One of ``'salt'``, ``'free_nhc'``, ``'co2_adduct'``, ``'bicarbonate'``,
            ``'unknown'``.

    Returns:
        ``(azolium_smiles, net_charge)``. ``net_charge`` is the formal charge of
        the returned cation — normally +1, and +2 for bis-NHCs, whose two
        centres are both protonated. It is exactly one greater than the charge
        ``to_mono_carbene`` returns for the same input; ``to_free_carbene``
        agrees with ``to_mono_carbene`` only for mono-NHCs. Pass it into
        downstream xTB calls as ``--chrg``.
    """
    if form not in ("salt", "free_nhc", "co2_adduct", "bicarbonate", "unknown"):
        raise ValueError(f"Unknown form label: {form!r}")

    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        warnings.warn(f"to_azolium: RDKit failed to parse input SMILES: {smiles}")
        return smiles, 1
    mol.UpdatePropertyCache(strict=False)
    Chem.GetSSSR(mol)

    if form == "unknown":
        warnings.warn(
            f"to_azolium: form='unknown' for SMILES {smiles!r} — returning largest "
            "fragment unchanged."
        )
        cation = _largest_fragment(mol)
        return Chem.MolToSmiles(cation), sum(
            a.GetFormalCharge() for a in cation.GetAtoms()
        )

    if form == "co2_adduct":
        cation = _strip_co2_adduct(mol)
    else:  # salt | bicarbonate | free_nhc
        cation = _largest_fragment(mol)

    # A genuine azolium salt has no divalent carbene C left; free NHCs and
    # ylide-notated pre-catalysts ([N+]=[C-], net-neutral despite the 'salt'
    # label) still do, and must be protonated to reach the azolium.
    if _locate_free_carbene_carbons(cation):
        cation = _protonate_carbene(cation)

    canon = Chem.MolToSmiles(cation)
    net = sum(a.GetFormalCharge() for a in cation.GetAtoms())

    check = Chem.MolFromSmiles(canon, sanitize=False)
    if check is None:
        warnings.warn(
            f"to_azolium: result unparseable for {smiles!r} (form={form}); "
            "falling back to the largest fragment."
        )
        fallback = _largest_fragment(mol)
        return Chem.MolToSmiles(fallback), sum(
            a.GetFormalCharge() for a in fallback.GetAtoms()
        )

    return canon, net
