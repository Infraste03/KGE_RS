"""
SharedEmbedding: the foundation of the alternate learning architecture.

This module defines a single, learnable matrix of entity embeddings that is
SHARED between Task A (TransE on the KG) and Task B (SASRec on user sequences).

This is the implementation of design choice D3 (hard sharing): there is exactly
ONE embedding matrix in memory. When Task A computes its loss and backpropagates,
the gradients update this matrix. When Task B does the same, gradients update
the SAME matrix. This is what couples the two tasks together.

It also handles design choice D5 (warm start): the embedding matrix can be
initialized from the weights pre-trained in Step 2 (TransE via PyKEEN) and
in Step 3 (SASRec via RecBole), instead of starting from random.

----------------------------------------------------------------------------
Where to put this file:
    src/models/shared_embedding.py

Or, given the directory naming convention you adopted:
    B2B_alternate_learning/models/shared_embedding.py

The path doesn't matter as long as it's importable from your other modules.
----------------------------------------------------------------------------
"""

import os
import logging
import torch
import torch.nn as nn
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# =========================================================================== #
# THE SharedEmbedding CLASS
# =========================================================================== #

class SharedEmbedding(nn.Module):
    """
    A single embedding matrix shared between Task A and Task B.

    Internally this is just a torch.nn.Embedding wrapped in a tiny class.
    The "magic" is not in the implementation: it's in HOW we use it.
    The same instance of SharedEmbedding will be passed to both TransE and
    SASRec, so that both models read from and write to the SAME parameters.

    Parameters
    ----------
    num_entities : int
        Total number of entities (items + machines + models + clients).
        For your B2B-Parts-Rec dataset this is 24,900.
    embedding_dim : int
        Dimension of each entity embedding.
        Will be set to 200 by default (matching what we discussed for TransE).
    padding_idx : int, optional
        If not None, the embedding at this index is forced to zero and will
        not be updated by gradient descent. SASRec needs a "padding token"
        for variable-length sequences, so this is useful in Task B.
        Default: None (no padding).

    Attributes
    ----------
    embedding : nn.Embedding
        The actual matrix of shape (num_entities, embedding_dim).
    """

    def __init__(self, num_entities: int, embedding_dim: int, padding_idx: int = None):
        super().__init__()

        self.num_entities = num_entities
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

        # The embedding matrix.
        # We use Xavier uniform initialization as a default. This will be
        # overwritten if the user calls one of the warm-start methods below.
        self.embedding = nn.Embedding(
            num_embeddings=num_entities,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )
        nn.init.xavier_uniform_(self.embedding.weight)

        # If we have a padding index, force it to zero AFTER xavier init.
        if padding_idx is not None:
            with torch.no_grad():
                self.embedding.weight[padding_idx].fill_(0.0)

        logger.info(
            f"Initialized SharedEmbedding: "
            f"{num_entities:,} entities x {embedding_dim} dims = "
            f"{num_entities * embedding_dim:,} parameters "
            f"(padding_idx={padding_idx})"
        )

    def forward(self, entity_ids: torch.Tensor) -> torch.Tensor:
        """
        Look up the embeddings for a batch of entity IDs.

        This is the only way Task A and Task B will access the embeddings.
        Whoever calls this, gets the embeddings; gradients flowing back
        will update self.embedding.weight.
        """
        return self.embedding(entity_ids)

    # ----------------------------------------------------------------------- #
    # Warm-start methods (design choice D5)
    # ----------------------------------------------------------------------- #

    def load_from_pykeen_transe(
        self,
        pykeen_weights: torch.Tensor,
        entity_id_mapping_pykeen: dict,
        entity_id_mapping_ours: dict,
        verbose: bool = True,
    ):
        """
        Load TransE entity embeddings trained with PyKEEN in Step 2.

        Why we need a "mapping translation":
            PyKEEN assigns its own integer IDs to entities (different from
            the IDs we assigned in load_kg.py). So we cannot just copy the
            matrix as-is: row i in PyKEEN's matrix may correspond to a
            different entity than row i in our matrix.

            For each of our entity IDs, we look up the entity string,
            find PyKEEN's ID for that same string, and copy the right row.

        Parameters
        ----------
        pykeen_weights : torch.Tensor of shape (num_entities, embedding_dim)
            The matrix you saved at the end of Step 2.
        entity_id_mapping_pykeen : dict[str, int]
            PyKEEN's mapping: entity_string -> PyKEEN's integer ID.
            You can extract this from PyKEEN's TriplesFactory after training.
        entity_id_mapping_ours : dict[str, int]
            Our mapping from load_kg.py: entity_string -> our integer ID.
        verbose : bool
            If True, print summary stats before/after loading.
        """
        if verbose:
            logger.info("\n--- Warm start: loading TransE weights from PyKEEN ---")

        # ---- Sanity check 1: the dimensions must match ----
        if pykeen_weights.shape[1] != self.embedding_dim:
            raise ValueError(
                f"PyKEEN embedding dim ({pykeen_weights.shape[1]}) "
                f"does not match our embedding dim ({self.embedding_dim}). "
                f"You must train TransE with the same dim you use here."
            )

        # ---- Sanity check 2: every entity in our mapping should exist in PyKEEN ----
        missing = [e for e in entity_id_mapping_ours if e not in entity_id_mapping_pykeen]
        if missing:
            raise ValueError(
                f"{len(missing)} entities are in our mapping but not in PyKEEN's. "
                f"Examples: {missing[:5]}. "
                f"This usually means the KG used in Step 2 is different from "
                f"the one loaded here."
            )

        # ---- Compute statistics BEFORE loading (for the diagnostic print) ----
        norm_before = self.embedding.weight.norm(dim=1).mean().item()

        # ---- Build the index mapping and copy the weights ----
        # For each entity, copy the corresponding PyKEEN row into our matrix.
        with torch.no_grad():
            for entity_str, our_id in entity_id_mapping_ours.items():
                pykeen_id = entity_id_mapping_pykeen[entity_str]
                self.embedding.weight[our_id] = pykeen_weights[pykeen_id]

            # Re-zero the padding index if we have one
            if self.padding_idx is not None:
                self.embedding.weight[self.padding_idx].fill_(0.0)

        # ---- Compute statistics AFTER loading ----
        norm_after = self.embedding.weight.norm(dim=1).mean().item()

        if verbose:
            logger.info(f"  Loaded weights for {len(entity_id_mapping_ours):,} entities")
            logger.info(f"  Mean L2 norm BEFORE warm start: {norm_before:.4f} (random init)")
            logger.info(f"  Mean L2 norm AFTER  warm start: {norm_after:.4f} (PyKEEN's TransE)")
            logger.info(
                f"  These should differ. If they are equal, the warm start did NOTHING."
            )

    def load_from_sasrec_recbole(
        self,
        sasrec_item_embeddings: torch.Tensor,
        recbole_item_id_mapping: dict,
        entity_id_mapping_ours: dict,
        verbose: bool = True,
    ):
        """
        Load SASRec item embeddings trained with RecBole in Step 3.

        Important detail (D5 design choice):
            We discussed three options for warm-starting from BOTH TransE
            and SASRec at the same time:
                (1) Use only TransE's weights as init  <--- our default
                (2) Use only SASRec's weights as init
                (3) Average the two (or some mix)

            We chose option (1): we initialize from TransE, and we ignore
            SASRec's item embeddings here. The reason is that the KG carries
            structural information (compatibility) that we want preserved
            in the shared matrix. SASRec's self-attention weights ARE still
            warm-started: just not via the SharedEmbedding, but via the
            self-attention layers themselves (handled in task_b_sasrec.py).

        This method is provided for completeness, in case we want to ablate
        and try option (2) or (3) later. By default it is NOT called in
        run_step4.py.

        Parameters
        ----------
        sasrec_item_embeddings : torch.Tensor of shape (num_items_recbole, dim)
            The item embedding matrix from your saved SASRec checkpoint.
            From your test_performance.py you can extract this with:
                state_dict = torch.load(checkpoint_path)['state_dict']
                sasrec_item_embeddings = state_dict['item_embedding.weight']
        recbole_item_id_mapping : dict[str, int]
            RecBole's mapping: item_string (e.g., 'item_1234') -> RecBole's int ID.
        entity_id_mapping_ours : dict[str, int]
            Our mapping: entity_string -> our integer ID.
        """
        if verbose:
            logger.info("\n--- Warm start: loading SASRec weights from RecBole ---")
            logger.info("  WARNING: this will OVERWRITE the items' rows in SharedEmbedding.")
            logger.info("  Use this only for ablations of D5. Default is to NOT call this.")

        if sasrec_item_embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"SASRec embedding dim ({sasrec_item_embeddings.shape[1]}) "
                f"does not match our embedding dim ({self.embedding_dim})."
            )

        # SASRec only has embeddings for items, not machines/models/clients.
        # So we only update rows corresponding to items.
        with torch.no_grad():
            n_loaded = 0
            for entity_str, our_id in entity_id_mapping_ours.items():
                if not entity_str.startswith('item_'):
                    continue
                if entity_str not in recbole_item_id_mapping:
                    continue
                recbole_id = recbole_item_id_mapping[entity_str]
                self.embedding.weight[our_id] = sasrec_item_embeddings[recbole_id]
                n_loaded += 1

            if self.padding_idx is not None:
                self.embedding.weight[self.padding_idx].fill_(0.0)

        if verbose:
            logger.info(f"  Loaded SASRec weights for {n_loaded:,} item entities")

    # ----------------------------------------------------------------------- #
    # Diagnostics
    # ----------------------------------------------------------------------- #

    def diagnostic_summary(self) -> dict:
        """
        Compute simple statistics about the embedding matrix.

        Useful before and after warm start, and at every epoch during training,
        to make sure the matrix is "alive" (gradients are flowing, norms are
        sensible, no NaN, etc.).
        """
        with torch.no_grad():
            w = self.embedding.weight
            norms = w.norm(dim=1)
            stats = {
                'shape': tuple(w.shape),
                'norm_mean': norms.mean().item(),
                'norm_std': norms.std().item(),
                'norm_min': norms.min().item(),
                'norm_max': norms.max().item(),
                'has_nan': torch.isnan(w).any().item(),
                'has_inf': torch.isinf(w).any().item(),
                'requires_grad': w.requires_grad,
            }
        return stats

    def print_diagnostic(self, label: str = ""):
        """Print the diagnostic summary in a readable format."""
        s = self.diagnostic_summary()
        logger.info(f"\n--- SharedEmbedding diagnostic [{label}] ---")
        logger.info(f"  shape         : {s['shape']}")
        logger.info(f"  norm (L2) mean: {s['norm_mean']:.4f}")
        logger.info(f"  norm (L2) std : {s['norm_std']:.4f}")
        logger.info(f"  norm (L2) min : {s['norm_min']:.4f}")
        logger.info(f"  norm (L2) max : {s['norm_max']:.4f}")
        logger.info(f"  has NaN       : {s['has_nan']}")
        logger.info(f"  has Inf       : {s['has_inf']}")
        logger.info(f"  requires_grad : {s['requires_grad']}")
        if s['has_nan'] or s['has_inf']:
            logger.error("  PROBLEM: NaN or Inf found in embedding weights!")


# =========================================================================== #
# STANDALONE SMOKE TEST
#
# When you run `python shared_embedding.py` directly, this block runs.
# It builds a small SharedEmbedding from scratch and verifies the basics.
# =========================================================================== #

if __name__ == "__main__":

    print("=" * 70)
    print("SHARED EMBEDDING SMOKE TEST")
    print("=" * 70)

    # ---- Build a small SharedEmbedding (using the real B2B-Parts-Rec sizes) ----
    NUM_ENTITIES = 24_900
    EMBEDDING_DIM = 200

    shared = SharedEmbedding(
        num_entities=NUM_ENTITIES,
        embedding_dim=EMBEDDING_DIM,
        padding_idx=None,  # SASRec will pass padding_idx=0 later if needed
    )

    # ---- Diagnostic at random init ----
    shared.print_diagnostic(label="after random init")

    # ---- Test the forward pass: look up some entities ----
    test_ids = torch.LongTensor([0, 100, 1000, 24899])
    out = shared(test_ids)
    print(f"\nForward pass test:")
    print(f"  input shape : {test_ids.shape}")
    print(f"  output shape: {out.shape}")
    assert out.shape == (4, EMBEDDING_DIM), "Forward pass shape mismatch!"

    # ---- Test that gradients flow ----
    out.sum().backward()
    grad = shared.embedding.weight.grad
    assert grad is not None, "Gradients did not flow!"
    print(f"  gradient shape: {grad.shape}")
    print(f"  gradient is non-zero on the touched rows: {(grad[test_ids].abs().sum(dim=1) > 0).all().item()}")

    # ---- Test the warm-start sanity checks ----
    print("\n--- Testing warm-start dimension check ---")
    fake_pykeen_weights = torch.randn(NUM_ENTITIES, EMBEDDING_DIM)
    fake_entity_to_id_pykeen = {f"item_{i}": i for i in range(NUM_ENTITIES)}
    fake_entity_to_id_ours = {f"item_{i}": i for i in range(NUM_ENTITIES)}

    # This should succeed
    shared.load_from_pykeen_transe(
        pykeen_weights=fake_pykeen_weights,
        entity_id_mapping_pykeen=fake_entity_to_id_pykeen,
        entity_id_mapping_ours=fake_entity_to_id_ours,
        verbose=True,
    )

    # ---- Verify the load actually changed the weights ----
    norms_after = shared.embedding.weight.norm(dim=1).mean().item()
    print(f"\n  Mean norm after loading random PyKEEN weights: {norms_after:.4f}")
    print(f"  (Random PyKEEN weights have ~sqrt(dim)=14.1 norm on average)")

    # ---- Test the dimension mismatch error ----
    print("\n--- Testing that dimension mismatch is caught ---")
    bad_pykeen_weights = torch.randn(NUM_ENTITIES, EMBEDDING_DIM + 1)
    try:
        shared.load_from_pykeen_transe(
            pykeen_weights=bad_pykeen_weights,
            entity_id_mapping_pykeen=fake_entity_to_id_pykeen,
            entity_id_mapping_ours=fake_entity_to_id_ours,
            verbose=False,
        )
        print("  PROBLEM: dimension mismatch was NOT caught")
    except ValueError as e:
        print(f"  PASS: caught the dimension mismatch correctly")

    print("\n" + "=" * 70)
    print("SMOKE TEST PASSED")
    print("=" * 70)