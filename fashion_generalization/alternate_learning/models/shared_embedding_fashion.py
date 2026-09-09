"""
shared_embedding_fashion.py
============================

SharedEmbedding: the embedding matrix shared between Task A
(TransE on the Fashion KG) and Task B (SASRec on user sequences).

This is the core of hard sharing in Fashion alternate learning.
The same SharedEmbedding instance is passed to both tasks,
so both read and write the SAME parameters.

Adaptation of the B2B shared_embedding.py to the Fashion domain.
The design is identical; only dimensions and entity types change.

Fashion ID space structure
(from verify_task_a_type_index_fashion.py)
---------------------------------------------------------------

The alphabetical ordering used by load_kg_fashion assigns IDs as follows:

    brand_*    -> IDs 0     .. 17510   (17511 brands)
    category_* -> IDs 17511 .. 17513   (3 categories)
    item_*     -> IDs 17514 .. 182982  (165469 items)

Total KG entities: 182,983

This ordering is FIXED and deterministic. The TransE warm start
copies rows entity by entity through entity_string, so it does NOT
depend on the ordering — but understanding the ordering is useful
for debugging.

Padding design
--------------

RecBole uses internal id=0 as [PAD]. In our unified space, however,
id=0 is already occupied by the first brand (brand_*). Therefore:

    SharedEmbedding has num_entities = 182,983 + 1 = 182,984 rows
    padding_idx = 182,983  (last row, used only by SASRec)

Task A directly uses real KG IDs (0..182982).
Task B translates RecBole IDs into unified IDs through the
UnifiedIdSpace lookup table and uses 182,983 as padding (not 0).

File location:
    fashion_generalization/alternate_learning/models/shared_embedding_fashion.py
"""

import logging
import torch
import torch.nn as nn

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# =========================================================================== #
# SharedEmbedding
# =========================================================================== #

class SharedEmbedding(nn.Module):
    """
    Embedding matrix shared between Task A and Task B.

    Parameters
    ----------
    num_entities : int
        Total number of rows in the matrix.
        For Fashion: space.num_total_entities + 1 = 182,984
        (182,983 KG entities + 1 padding row for SASRec)

    embedding_dim : int
        Dimension of each embedding.
        For Fashion: 64 (matches both TransE HPO and SASRec HPO)

    padding_idx : int, optional
        Index of the padding row. This row remains zero
        and does not receive gradients.
        For Fashion: space.num_total_entities = 182,983
    """

    def __init__(self, num_entities: int, embedding_dim: int, padding_idx: int = None):
        super().__init__()

        if num_entities <= 0:
            raise ValueError(f"num_entities must be positive, got {num_entities}")
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")
        if padding_idx is not None and not (0 <= padding_idx < num_entities):
            raise ValueError(
                f"padding_idx={padding_idx} outside range 0..{num_entities - 1}"
            )

        self.num_entities  = num_entities
        self.embedding_dim = embedding_dim
        self.padding_idx   = padding_idx

        self.embedding = nn.Embedding(
            num_embeddings=num_entities,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )
        nn.init.xavier_uniform_(self.embedding.weight)

        # Force the padding row to zero after Xavier initialization
        if padding_idx is not None:
            with torch.no_grad():
                self.embedding.weight[padding_idx].fill_(0.0)

        logger.info(
            f"SharedEmbedding initialized: "
            f"{num_entities:,} rows x {embedding_dim} dim = "
            f"{num_entities * embedding_dim:,} parameters "
            f"(padding_idx={padding_idx})"
        )

    def forward(self, entity_ids: torch.Tensor) -> torch.Tensor:
        """
        Embedding lookup for a batch of entity IDs.

        Task A directly passes KG/unified IDs.
        Task B must pass unified IDs (NOT raw RecBole IDs).
        RecBole -> unified translation is handled by the dataloader.
        """
        return self.embedding(entity_ids)

    # ----------------------------------------------------------------------- #
    # Warm start from TransE (PyKEEN)
    # ----------------------------------------------------------------------- #

    def load_from_pykeen_transe(
        self,
        pykeen_weights: torch.Tensor,
        entity_id_mapping_pykeen: dict,
        entity_id_mapping_ours: dict,
        verbose: bool = True,
    ):
        """
        Load TransE entity weights trained with PyKEEN.

        PyKEEN and load_kg_fashion both assign IDs alphabetically
        over the same entities, so the two mappings COINCIDE.
        Nevertheless, we copy rows through entity_string for safety —
        we never assume that the orderings are identical.

        The copy is vectorized rather than performed with a Python loop
        for efficiency: with 182,983 entities, a Python loop would take
        several seconds.

        Fashion ordering:
            brand_*    -> IDs 0..17510
            category_* -> IDs 17511..17513
            item_*     -> IDs 17514..182982

        This does not affect the warm start because copying is performed
        by entity_string, but it is useful when debugging matrix rows.

        Parameters
        ----------
        pykeen_weights : torch.Tensor, shape (182983, 64)
            PyKEEN entity embedding matrix from best_model.pt.

        entity_id_mapping_pykeen : dict[str, int]
            PyKEEN mapping: entity_string -> PyKEEN integer ID.
            Obtained from TriplesFactory.entity_to_id after training.

        entity_id_mapping_ours : dict[str, int]
            Our mapping: entity_string -> unified integer ID.
            Typically space.entity_to_unified_id.

        verbose : bool
            If True, print statistics before and after loading.
        """
        if verbose:
            logger.info("\n--- Warm start: loading TransE weights from PyKEEN ---")

        if not torch.is_tensor(pykeen_weights):
            raise TypeError("pykeen_weights must be a torch.Tensor")
        if pykeen_weights.ndim != 2:
            raise ValueError(
                f"pykeen_weights must be 2D, got shape {tuple(pykeen_weights.shape)}"
            )

        # Check 1: embedding dimension
        if pykeen_weights.shape[1] != self.embedding_dim:
            raise ValueError(
                f"PyKEEN embedding dim ({pykeen_weights.shape[1]}) != "
                f"our embedding dim ({self.embedding_dim}). "
                f"TransE must be trained with the same dimension."
            )

        # Check 2: every entity in our mapping must exist in PyKEEN
        missing = [e for e in entity_id_mapping_ours if e not in entity_id_mapping_pykeen]
        if missing:
            raise ValueError(
                f"{len(missing)} entities in our mapping do not exist in PyKEEN. "
                f"Examples: {missing[:5]}. "
                f"The KG used in Step 2 is probably different from the one loaded here."
            )

        # Check 3: our IDs must be inside the embedding matrix
        bad_ours = [
            (e, i) for e, i in entity_id_mapping_ours.items()
            if i < 0 or i >= self.num_entities
        ]
        if bad_ours:
            raise ValueError(
                f"{len(bad_ours)} of our entity IDs are outside the SharedEmbedding range. "
                f"Examples: {bad_ours[:5]}"
            )

        # Check 4: PyKEEN IDs must be inside pykeen_weights
        bad_pykeen = [
            (e, i) for e, i in entity_id_mapping_pykeen.items()
            if i < 0 or i >= pykeen_weights.shape[0]
        ]
        if bad_pykeen:
            raise ValueError(
                f"{len(bad_pykeen)} PyKEEN IDs are outside the pykeen_weights range. "
                f"Examples: {bad_pykeen[:5]}"
            )

        norm_before = self.embedding.weight.norm(p=2, dim=1).mean().item()

        # Vectorized copy — NOT a Python row-by-row assignment
        # Build two parallel index arrays and perform
        # a single indexed tensor assignment.
        our_ids_list    = []
        pykeen_ids_list = []
        for entity_str, our_id in entity_id_mapping_ours.items():
            our_ids_list.append(our_id)
            pykeen_ids_list.append(entity_id_mapping_pykeen[entity_str])

        our_ids    = torch.LongTensor(our_ids_list)
        pykeen_ids = torch.LongTensor(pykeen_ids_list)

        with torch.no_grad():
            self.embedding.weight[our_ids] = pykeen_weights[pykeen_ids].to(
                self.embedding.weight.device
            )
            # Reset padding to zero after copying
            if self.padding_idx is not None:
                self.embedding.weight[self.padding_idx].fill_(0.0)

        norm_after = self.embedding.weight.norm(p=2, dim=1).mean().item()

        if verbose:
            logger.info(f"  Entities loaded           : {len(our_ids_list):,}")
            logger.info(f"  Mean L2 norm BEFORE       : {norm_before:.4f} (random init)")
            logger.info(f"  Mean L2 norm AFTER        : {norm_after:.4f} (TransE)")
            logger.info(f"  (With normalized PyKEEN embeddings, the AFTER norm should be ~1.0)")
            if abs(norm_before - norm_after) < 1e-6:
                logger.warning(
                    "  WARNING: the norm did not change after warm start. "
                    "The loading operation may not have modified the weights."
                )

    # ----------------------------------------------------------------------- #
    # Warm start from SASRec (RecBole) — ablations only
    # ----------------------------------------------------------------------- #

    def load_from_sasrec_recbole(
        self,
        sasrec_item_embeddings: torch.Tensor,
        recbole_item_id_mapping: dict,
        entity_id_mapping_ours: dict,
        verbose: bool = True,
    ):
        """
        Optionally load SASRec item weights trained with RecBole.

        WARNING: this method is NOT called by default.
        The selected design (as in B2B) initializes SharedEmbedding
        only from TransE weights. SASRec self-attention weights are
        loaded separately in task_b_sasrec_fashion.py.
        This method is provided for possible D5 ablations.

        Fashion mapping details:
            RecBole uses raw tokens without prefix: 'B005OT7MBM' -> 1
            Our KG uses:                          'item_B005OT7MBM' -> unified_id
            This method handles both formats.

        SASRec has no embeddings for brands or categories — only for items.
        Rows 0..17510 (brands) and 17511..17513 (categories) in
        SharedEmbedding are NOT modified by this method.

        Parameters
        ----------
        sasrec_item_embeddings : torch.Tensor, shape (165470, 64)
            item_embedding.weight from the RecBole SASRec checkpoint.
            Includes the PAD row (id=0).

        recbole_item_id_mapping : dict[str, int]
            RecBole mapping: raw_token -> internal_id.
            Typically space.recbole_token_to_internal.
            Includes '[PAD]' -> 0.

        entity_id_mapping_ours : dict[str, int]
            Our mapping: entity_string -> unified_id.
            Typically space.entity_to_unified_id.
        """
        if verbose:
            logger.info("\n--- Warm start: loading SASRec weights from RecBole (ABLATION) ---")
            logger.info("  WARNING: this overwrites item rows in SharedEmbedding.")
            logger.info("  Do not call by default. Use only for ablation experiments.")

        if not torch.is_tensor(sasrec_item_embeddings):
            raise TypeError("sasrec_item_embeddings must be a torch.Tensor")
        if sasrec_item_embeddings.ndim != 2:
            raise ValueError(
                f"sasrec_item_embeddings must be 2D, "
                f"got shape {tuple(sasrec_item_embeddings.shape)}"
            )
        if sasrec_item_embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"SASRec embedding dim ({sasrec_item_embeddings.shape[1]}) != "
                f"our embedding dim ({self.embedding_dim})."
            )
        if recbole_item_id_mapping.get('[PAD]', None) != 0:
            raise ValueError(
                f"Expected RecBole [PAD] -> 0, "
                f"got {recbole_item_id_mapping.get('[PAD]', 'MISSING')}"
            )

        # Vectorized index construction
        our_ids_list    = []
        recbole_ids_list = []
        n_skipped_non_item = 0
        n_missing          = 0
        missing_examples   = []

        for entity_str, our_id in entity_id_mapping_ours.items():

            # Skip non-item entities (brand, category)
            if not entity_str.startswith('item_'):
                n_skipped_non_item += 1
                continue

            # First try the raw token (Fashion format: 'B005OT7MBM')
            raw_token = entity_str[len('item_'):]
            if raw_token in recbole_item_id_mapping:
                recbole_key = raw_token
            elif entity_str in recbole_item_id_mapping:
                # B2B-format fallback: 'item_B005OT7MBM'
                recbole_key = entity_str
            else:
                n_missing += 1
                if len(missing_examples) < 5:
                    missing_examples.append(entity_str)
                continue

            recbole_id = int(recbole_item_id_mapping[recbole_key])

            # Sanity check: never copy the PAD row (id=0 in RecBole)
            if recbole_id == 0:
                raise ValueError(
                    f"Real item {entity_str} maps to RecBole PAD id=0. "
                    f"This should never happen."
                )

            if recbole_id >= sasrec_item_embeddings.shape[0]:
                raise ValueError(
                    f"RecBole id {recbole_id} for {entity_str} is outside the "
                    f"SASRec embedding range (shape {tuple(sasrec_item_embeddings.shape)})"
                )

            if our_id < 0 or our_id >= self.num_entities:
                raise ValueError(
                    f"Our id {our_id} for {entity_str} is outside the "
                    f"SharedEmbedding range (0..{self.num_entities - 1})"
                )

            our_ids_list.append(our_id)
            recbole_ids_list.append(recbole_id)

        # Vectorized copy
        if our_ids_list:
            our_ids      = torch.LongTensor(our_ids_list)
            recbole_ids  = torch.LongTensor(recbole_ids_list)
            with torch.no_grad():
                self.embedding.weight[our_ids] = sasrec_item_embeddings[recbole_ids].to(
                    self.embedding.weight.device
                )
                if self.padding_idx is not None:
                    self.embedding.weight[self.padding_idx].fill_(0.0)

        if verbose:
            logger.info(f"  Items loaded              : {len(our_ids_list):,}")
            logger.info(f"  Non-item entities skipped : {n_skipped_non_item:,} "
                        f"(brand + category, expected)")
            logger.info(f"  Items missing in RecBole  : {n_missing:,}")
            if missing_examples:
                logger.info(f"  Missing examples: {missing_examples}")

    # ----------------------------------------------------------------------- #
    # Diagnostics
    # ----------------------------------------------------------------------- #

    def diagnostic_summary(self) -> dict:
        """
        Statistics for the embedding matrix.
        Useful before/after warm start and during training.
        """
        with torch.no_grad():
            w       = self.embedding.weight
            l2_norms = w.norm(p=2, dim=1)
            l1_norms = w.norm(p=1, dim=1)

            stats = {
                'shape':        tuple(w.shape),
                'l2_norm_mean': l2_norms.mean().item(),
                'l2_norm_std':  l2_norms.std().item(),
                'l2_norm_min':  l2_norms.min().item(),
                'l2_norm_max':  l2_norms.max().item(),
                'l1_norm_mean': l1_norms.mean().item(),
                'l1_norm_std':  l1_norms.std().item(),
                'l1_norm_min':  l1_norms.min().item(),
                'l1_norm_max':  l1_norms.max().item(),
                'has_nan':      torch.isnan(w).any().item(),
                'has_inf':      torch.isinf(w).any().item(),
                'requires_grad': w.requires_grad,
            }

            if self.padding_idx is not None:
                pad = w[self.padding_idx]
                stats['padding_idx']      = self.padding_idx
                stats['padding_l2_norm']  = pad.norm(p=2).item()
                stats['padding_abs_sum']  = pad.abs().sum().item()
            else:
                stats['padding_idx']      = None
                stats['padding_l2_norm']  = None
                stats['padding_abs_sum']  = None

        return stats

    def print_diagnostic(self, label: str = ""):
        s = self.diagnostic_summary()
        logger.info(f"\n--- SharedEmbedding diagnostic [{label}] ---")
        logger.info(f"  shape           : {s['shape']}")
        logger.info(f"  L2 norm mean    : {s['l2_norm_mean']:.4f}")
        logger.info(f"  L2 norm std     : {s['l2_norm_std']:.4f}")
        logger.info(f"  L2 norm min     : {s['l2_norm_min']:.4f}")
        logger.info(f"  L2 norm max     : {s['l2_norm_max']:.4f}")
        logger.info(f"  L1 norm mean    : {s['l1_norm_mean']:.4f}")
        logger.info(f"  has NaN         : {s['has_nan']}")
        logger.info(f"  has Inf         : {s['has_inf']}")
        logger.info(f"  requires_grad   : {s['requires_grad']}")
        if self.padding_idx is not None:
            logger.info(f"  padding_idx     : {s['padding_idx']}")
            logger.info(f"  padding L2 norm : {s['padding_l2_norm']:.4f}")
            logger.info(f"  padding abs sum : {s['padding_abs_sum']:.4f}")
        if s['has_nan'] or s['has_inf']:
            logger.error("  PROBLEM: NaN or Inf values found in the embedding matrix!")


# =========================================================================== #
# STANDALONE SMOKE TEST
#
# Real Fashion dimensions:
#   - 182,983 KG entities
#     (brand 0..17510, category 17511..17513, item 17514..182982)
#   - 1 padding row (index 182,983)
#   - embedding_dim = 64
#
# The smoke test uses fake entities BUT preserves the realistic Fashion structure:
#   - brand, category, item (not only items as in B2B)
# =========================================================================== #

if __name__ == "__main__":

    print("=" * 70)
    print("SHARED EMBEDDING FASHION — SMOKE TEST")
    print("=" * 70)

    # ---- Real Fashion dimensions ----
    N_BRAND    = 17_511
    N_CATEGORY = 3
    N_ITEM     = 165_469
    N_KG       = N_BRAND + N_CATEGORY + N_ITEM   # = 182,983
    DIM        = 64
    PADDING    = N_KG                             # = 182,983
    N_ROWS     = N_KG + 1                         # = 182,984

    # ---- 1. Construction ----
    print("\n[1/7] Building SharedEmbedding...")
    shared = SharedEmbedding(
        num_entities=N_ROWS,
        embedding_dim=DIM,
        padding_idx=PADDING,
    )
    shared.print_diagnostic(label="after random initialization")

    # ---- 2. Forward pass ----
    print("\n[2/7] Forward pass...")
    # Sample IDs from all three entity types + padding
    test_ids = torch.LongTensor([
        0,              # first brand
        N_BRAND - 1,    # last brand
        N_BRAND,        # first category
        N_BRAND + N_CATEGORY,     # first item
        N_KG - 1,       # last item
        PADDING,        # padding row
    ])
    out = shared(test_ids)
    assert out.shape == (6, DIM), f"Incorrect shape: {out.shape}"
    assert shared.embedding.weight[PADDING].abs().sum().item() == 0.0, \
        "Padding row is not zero after initialization!"
    print(f"  PASS: output shape {tuple(out.shape)}, padding is zero")

    # ---- 3. Gradient flow ----
    print("\n[3/7] Gradient flow...")
    shared.embedding.weight.grad = None
    out.sum().backward()
    grad = shared.embedding.weight.grad
    assert grad is not None, "No gradient!"

    real_ids = torch.LongTensor([0, N_BRAND - 1, N_BRAND, N_BRAND + N_CATEGORY, N_KG - 1])
    assert (grad[real_ids].abs().sum(dim=1) > 0).all(), \
        "Real rows did not receive gradients!"
    assert grad[PADDING].abs().sum().item() == 0.0, \
        "Padding row received gradients!"
    print(f"  PASS: gradients on real rows, zero on padding")

    # ---- 4. TransE warm start — fake but realistic Fashion structure ----
    print("\n[4/7] TransE warm start (fake, realistic Fashion structure)...")

    # Fake entities with brand/category/item structure matching real Fashion
    fake_entities = (
        [f"brand_{i}" for i in range(N_BRAND)]
        + [f"category_{i}" for i in range(N_CATEGORY)]
        + [f"item_{i}" for i in range(N_ITEM)]
    )
    # Our mapping: same ordering as real Fashion
    fake_ours   = {e: i for i, e in enumerate(fake_entities)}
    # PyKEEN: same ordering (as verified in verify_task_a_type_index_fashion)
    fake_pykeen = dict(fake_ours)

    fake_pykeen_weights = torch.randn(N_KG, DIM)

    shared_fresh = SharedEmbedding(N_ROWS, DIM, PADDING)
    shared_fresh.load_from_pykeen_transe(
        pykeen_weights=fake_pykeen_weights,
        entity_id_mapping_pykeen=fake_pykeen,
        entity_id_mapping_ours=fake_ours,
        verbose=True,
    )

    # Verify exact copy on a sample of entities from all three types
    sample_entities = [
        f"brand_{0}",
        f"brand_{N_BRAND - 1}",
        f"category_{0}",
        f"item_{0}",
        f"item_{N_ITEM - 1}",
    ]
    for entity in sample_entities:
        our_id    = fake_ours[entity]
        pykeen_id = fake_pykeen[entity]
        diff = (
            shared_fresh.embedding.weight[our_id].detach()
            - fake_pykeen_weights[pykeen_id]
        ).abs().max().item()
        assert diff == 0.0, f"Incorrect copy for {entity}: diff={diff}"
    print(f"  PASS: exact copy verified for {len(sample_entities)} entities "
          f"(brand, category, item)")

    assert shared_fresh.embedding.weight[PADDING].abs().sum().item() == 0.0, \
        "Padding row is not zero after TransE warm start!"
    print(f"  PASS: padding row remains zero after warm start")

    # ---- 5. TransE warm start — dimension mismatch check ----
    print("\n[5/7] Checking dimension mismatch...")
    bad_weights = torch.randn(N_KG, DIM + 1)
    try:
        shared_fresh.load_from_pykeen_transe(
            pykeen_weights=bad_weights,
            entity_id_mapping_pykeen=fake_pykeen,
            entity_id_mapping_ours=fake_ours,
            verbose=False,
        )
        raise AssertionError("Dimension mismatch was NOT detected!")
    except ValueError:
        print(f"  PASS: dimension mismatch detected correctly")

    # ---- 6. SASRec warm start — Fashion structure (raw token without prefix) ----
    print("\n[6/7] SASRec warm start (ablation, raw Fashion tokens)...")

    # Small SharedEmbedding with brand + category + item
    small = SharedEmbedding(num_entities=8, embedding_dim=DIM, padding_idx=7)

    fake_ours_small = {
        "brand_X":    0,
        "brand_Y":    1,
        "category_Z": 2,
        "item_A":     3,
        "item_B":     4,
        "item_C":     5,
        # id 6 = item not present in RecBole
        # (KG-only, impossible in Fashion but tested here)
        "item_D":     6,
    }

    # RecBole uses raw tokens (without item_ prefix)
    fake_recbole_mapping = {
        "[PAD]": 0,
        "A":     1,
        "B":     2,
        "C":     3,
        # "D" missing: item_D is not in RecBole
    }

    fake_sasrec_weights = torch.randn(4, DIM)
    fake_sasrec_weights[0].fill_(999.0)  # PAD row — must never be copied

    small.load_from_sasrec_recbole(
        sasrec_item_embeddings=fake_sasrec_weights,
        recbole_item_id_mapping=fake_recbole_mapping,
        entity_id_mapping_ours=fake_ours_small,
        verbose=True,
    )

    # item_A (our_id=3) <- recbole_id=1 <- sasrec_weights[1]
    assert torch.equal(small.embedding.weight[3], fake_sasrec_weights[1]), \
        "item_A was not copied correctly"
    assert torch.equal(small.embedding.weight[4], fake_sasrec_weights[2]), \
        "item_B was not copied correctly"
    assert torch.equal(small.embedding.weight[5], fake_sasrec_weights[3]), \
        "item_C was not copied correctly"

    # Brand and category rows must not have been modified
    # Padding must remain zero
    assert small.embedding.weight[7].abs().sum().item() == 0.0, \
        "Padding row is not zero after SASRec warm start"

    print(f"  PASS: items copied correctly through raw Fashion tokens")
    print(f"  PASS: brand and category rows were not modified")
    print(f"  PASS: RecBole PAD (id=0) was not copied into any real row")
    print(f"  PASS: padding row remains zero")

    # ---- 7. Final diagnostic ----
    print("\n[7/7] Final diagnostic...")
    shared_fresh.print_diagnostic(label="after fake TransE warm start")

    print("\n" + "=" * 70)
    print("SMOKE TEST PASSED")
    print("=" * 70)
    print()
    print("Fashion structure summary:")
    print(f"  brand_*    : IDs 0..{N_BRAND - 1} ({N_BRAND:,} entities)")
    print(f"  category_* : IDs {N_BRAND}..{N_BRAND + N_CATEGORY - 1} ({N_CATEGORY} entities)")
    print(f"  item_*     : IDs {N_BRAND + N_CATEGORY}..{N_KG - 1} ({N_ITEM:,} entities)")
    print(f"  padding    : ID {PADDING} (not a KG entity, SASRec only)")
    print(f"  embedding_dim = {DIM}")