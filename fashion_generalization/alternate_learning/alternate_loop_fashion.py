"""
alternate_loop_fashion.py
==========================

Alternate training loop for Step 4 — Fashion domain.

Three scheduling modes (identical to B2B):
    'one_to_one' : one Task A batch, one Task B batch, alternating 1:1 (default)
    'epoch'      : one full Task A epoch, followed by one full Task B epoch
    'adaptive'   : the Task A:Task B ratio adapts according to the losses

Differences compared with B2B (alternate_loop.py):

    __init__:
      - Fashion NegativeSamplerRec uses item_id_start/item_id_end
        instead of num_items
      - There is no padding_idx to extract from model.task_b
        (Fashion uses unified_padding_idx, which is not required by the sampler
        because the sampling range is already correct)
      - margin_a default = 8.684
        (from Fashion TransE HPO, instead of 5.19 in B2B)

    _train_step_b:
      - pos_items and neg_items are already unified IDs (17514..182982)
        because the materialized DataLoader has already performed the translation
      - shared_embedding(pos_items) can therefore be used directly
      - Structurally identical to B2B: dot product user_repr * item_emb

    Smoke test:
      - item IDs are sampled in [ITEM_ID_START..ITEM_ID_END]
        instead of [1..num_items]
      - Verifies that SharedEmbedding receives gradients from both tasks
      - Verifies that SharedEmbedding weights change after training

File location:
    fashion_generalization/alternate_learning/alternate_loop_fashion.py
"""

import os
import sys
import logging
import time
import torch
from torch.utils.data import DataLoader, TensorDataset

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# This file is located in:
#   alternate_learning/alternate_loop_fashion.py
# while model components are located in:
#   alternate_learning/models/
MODELS_DIR = os.path.join(THIS_DIR, "models")

if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

if MODELS_DIR not in sys.path:
    sys.path.insert(0, MODELS_DIR)

from joint_model_fashion import JointAlternateModelFashion
from losses_fashion import TransELoss, SASRecBPRLoss
from samplers_fashion import NegativeSamplerKGE, NegativeSamplerRec

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Fashion constants
ITEM_ID_START = 17_514
ITEM_ID_END   = 182_982
NUM_KG_ENT    = 182_983


class AlternateTrainerFashion:
    """
    Alternate training of Task A (TransE) and Task B (SASRec)
    on the same JointAlternateModelFashion.

    Parameters
    ----------
    model : JointAlternateModelFashion
        Model with an already warm-started SharedEmbedding.
    loader_a : DataLoader
        Batch = (h, r, t) LongTensor, unified IDs [0..182982].
    loader_b : DataLoader
        Batch = (item_seq, item_seq_len, pos_items) LongTensor.
        item_seq: unified IDs, padding=182983.
        pos_items: unified IDs [17514..182982].
        Produced by build_rec_dataloader_fashion (already translated).
    num_entities : int
        Number of KG entities used by NegativeSamplerKGE.
        Fashion: 182983.
    learning_rate : float
        Adam learning rate.
    margin_a : float
        TransE loss margin. Default 8.684 from Fashion HPO.
    scheduling : str
        'one_to_one', 'epoch', or 'adaptive'.
    device : str
        'cuda' or 'cpu'.
    log_every : int
        Log every N batches.
    num_neg_b : int
        Number of negatives per positive for Task B.
        Default 1 (BPR).
    """

    def __init__(
        self,
        model: JointAlternateModelFashion,
        loader_a: DataLoader,
        loader_b: DataLoader,
        num_entities: int,
        learning_rate: float = 1e-3,
        margin_a: float = 8.684,
        scheduling: str = 'one_to_one',
        device: str = 'cpu',
        log_every: int = 100,
        num_neg_b: int = 1,
    ):
        if scheduling not in ('one_to_one', 'epoch', 'adaptive'):
            raise ValueError(
                f"scheduling must be one of 'one_to_one', 'epoch', 'adaptive'; "
                f"got '{scheduling}'"
            )

        self.model      = model.to(device)
        self.loader_a   = loader_a
        self.loader_b   = loader_b
        self.device     = device
        self.scheduling = scheduling
        self.log_every  = log_every
        self.num_neg_b  = num_neg_b

        # Single optimizer over all model parameters
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Loss functions
        self.loss_a_fn = TransELoss(margin=margin_a)
        self.loss_b_fn = SASRecBPRLoss()

        # Negative samplers
        # Task A: corrupt entities in [0..num_entities-1].
        # Padding (182983) is excluded by construction.
        self.sampler_a = NegativeSamplerKGE(num_entities=num_entities)

        # Task B: sample items in [ITEM_ID_START..ITEM_ID_END].
        # B2B DIFFERENCE: different interface — no num_items/padding_idx.
        self.sampler_b = NegativeSamplerRec(
            item_id_start=ITEM_ID_START,
            item_id_end=ITEM_ID_END,
        )

        # EMA memory for adaptive scheduling
        self._last_loss_a = 1.0
        self._last_loss_b = 1.0

        logger.info(
            f"AlternateTrainerFashion: scheduling={scheduling}, "
            f"lr={learning_rate}, margin_a={margin_a}, device={device}, "
            f"num_neg_b={num_neg_b}"
        )

    # --------------------------------------------------------------------- #
    # Individual training steps
    # --------------------------------------------------------------------- #

    def _train_step_a(self, batch_a) -> float:
        """
        Train one Task A (TransE) batch and return the loss as float.

        batch_a = (h, r, t) in unified IDs.
        Identical to B2B.
        """
        h, r, t = (x.to(self.device) for x in batch_a)
        h_neg, r_neg, t_neg = self.sampler_a.sample(h, r, t)

        pos_scores = self.model.forward_kge(h, r, t)
        neg_scores = self.model.forward_kge(h_neg, r_neg, t_neg)

        loss = self.loss_a_fn(pos_scores, neg_scores)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def _train_step_b(self, batch_b) -> float:
        """
        Train one Task B (SASRec BPR) batch and return the loss as float.

        batch_b = (item_seq, item_seq_len, pos_items) in unified IDs.
        pos_items are already unified IDs [17514..182982], so no conversion
        is required.

        DIFFERENCE compared with B2B:
        In B2B, pos_items were RecBole IDs and shared_embedding used them
        directly as indices. In Fashion, pos_items are already the correct
        unified IDs.

        The code structure remains the same because shared_embedding accepts
        any LongTensor containing valid indices.
        """
        self.optimizer.zero_grad()

        item_seq, item_seq_len, pos_items = (
            x.to(self.device) for x in batch_b
        )

        # User representation
        user_repr = self.model.forward_sasrec(
            item_seq,
            item_seq_len
        )  # (B, 64)

        # Positive item embedding from SharedEmbedding
        pos_emb = self.model.shared_embedding(
            pos_items
        )  # (B, 64)

        # Sample negatives in unified ID space
        # [ITEM_ID_START..ITEM_ID_END]
        neg_items = self.sampler_b.sample(
            pos_items,
            num_neg=self.num_neg_b
        )

        neg_items = neg_items.to(self.device)

        neg_emb = self.model.shared_embedding(
            neg_items
        )  # (B, num_neg, 64) or (B, 64)

        # Scores: dot product user_repr * item_emb
        pos_score = (
            user_repr * pos_emb
        ).sum(dim=-1)  # (B,)

        neg_score = (
            user_repr.unsqueeze(1) * neg_emb
        ).sum(dim=-1)  # (B, num_neg)

        loss = self.loss_b_fn(
            pos_score.unsqueeze(1),
            neg_score
        )

        loss.backward()
        self.optimizer.step()

        return loss.item()

    # --------------------------------------------------------------------- #
    # Single-epoch loops
    # --------------------------------------------------------------------- #

    def _epoch_one_to_one(self):
        """1:1 — one A batch, one B batch, until both loaders are exhausted."""
        iter_a = iter(self.loader_a)
        iter_b = iter(self.loader_b)

        losses_a, losses_b = [], []
        step = 0

        exhausted_a = exhausted_b = False

        while not (exhausted_a and exhausted_b):

            if not exhausted_a:
                try:
                    loss_a = self._train_step_a(
                        next(iter_a)
                    )
                    losses_a.append(loss_a)
                except StopIteration:
                    exhausted_a = True

            if not exhausted_b:
                try:
                    loss_b = self._train_step_b(
                        next(iter_b)
                    )
                    losses_b.append(loss_b)
                except StopIteration:
                    exhausted_b = True

            step += 1

            if step % self.log_every == 0:
                la = (
                    losses_a[-1]
                    if losses_a
                    else float('nan')
                )

                lb = (
                    losses_b[-1]
                    if losses_b
                    else float('nan')
                )

                logger.info(
                    f"  step {step}  "
                    f"loss_a={la:.4f}  "
                    f"loss_b={lb:.4f}"
                )

        return losses_a, losses_b

    def _epoch_epoch(self):
        """Run one full Task A epoch followed by one full Task B epoch."""
        losses_a, losses_b = [], []

        logger.info("  [Task A phase]")

        for i, batch_a in enumerate(self.loader_a):
            loss_a = self._train_step_a(batch_a)
            losses_a.append(loss_a)

            if (i + 1) % self.log_every == 0:
                logger.info(
                    f"    A step {i+1}  "
                    f"loss_a={loss_a:.4f}"
                )

        logger.info("  [Task B phase]")

        for i, batch_b in enumerate(self.loader_b):
            loss_b = self._train_step_b(batch_b)
            losses_b.append(loss_b)

            if (i + 1) % self.log_every == 0:
                logger.info(
                    f"    B step {i+1}  "
                    f"loss_b={loss_b:.4f}"
                )

        return losses_a, losses_b

    def _epoch_adaptive(self):
        """Adaptive scheduling — the task with higher EMA loss receives more batches."""
        iter_a = iter(self.loader_a)
        iter_b = iter(self.loader_b)

        losses_a, losses_b = [], []

        exhausted_a = exhausted_b = False
        step = 0

        while not (exhausted_a and exhausted_b):

            do_a = (
                self._last_loss_a >= self._last_loss_b
            ) and not exhausted_a

            if exhausted_a:
                do_a = False

            elif exhausted_b:
                do_a = True

            # Constraint: at least one step out of three goes to Task B
            if step % 3 == 2 and not exhausted_b:
                do_a = False

            if do_a:
                try:
                    loss_a = self._train_step_a(
                        next(iter_a)
                    )

                    losses_a.append(loss_a)

                    self._last_loss_a = (
                        0.9 * self._last_loss_a
                        + 0.1 * loss_a
                    )

                except StopIteration:
                    exhausted_a = True
                    continue

            else:
                try:
                    loss_b = self._train_step_b(
                        next(iter_b)
                    )

                    losses_b.append(loss_b)

                    self._last_loss_b = (
                        0.9 * self._last_loss_b
                        + 0.1 * loss_b
                    )

                except StopIteration:
                    exhausted_b = True
                    continue

            step += 1

            if step % self.log_every == 0:
                logger.info(
                    f"  step {step}  "
                    f"ema_a={self._last_loss_a:.4f}  "
                    f"ema_b={self._last_loss_b:.4f}  "
                    f"done_a={len(losses_a)}  "
                    f"done_b={len(losses_b)}"
                )

        return losses_a, losses_b

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #

    def train_one_epoch(self) -> dict:
        """Run one epoch using the selected scheduling mode and return metrics."""
        self.model.train()
        t0 = time.time()

        if self.scheduling == 'one_to_one':
            losses_a, losses_b = self._epoch_one_to_one()

        elif self.scheduling == 'epoch':
            losses_a, losses_b = self._epoch_epoch()

        else:
            losses_a, losses_b = self._epoch_adaptive()

        dt = time.time() - t0

        return {
            'loss_a_mean': sum(losses_a) / max(len(losses_a), 1),
            'loss_b_mean': sum(losses_b) / max(len(losses_b), 1),
            'batches_a': len(losses_a),
            'batches_b': len(losses_b),
            'time_sec': dt,
        }

    def train(self, num_epochs: int) -> list:
        """Run num_epochs epochs and return one metrics dictionary per epoch."""
        history = []

        for epoch in range(1, num_epochs + 1):

            logger.info(
                f"\n=== Epoch {epoch}/{num_epochs} ==="
            )

            metrics = self.train_one_epoch()
            metrics['epoch'] = epoch

            history.append(metrics)

            logger.info(
                f"Epoch {epoch} done in "
                f"{metrics['time_sec']:.1f}s  "
                f"loss_a={metrics['loss_a_mean']:.4f}  "
                f"loss_b={metrics['loss_b_mean']:.4f}  "
                f"batches_a={metrics['batches_a']}  "
                f"batches_b={metrics['batches_b']}"
            )

        return history


# =========================================================================== #
# SMOKE TEST
# =========================================================================== #

def _make_fake_loaders_fashion(
    batch_size,
    n_triples=512,
    n_seqs=256,
    seq_len=10
):
    """
    Create DataLoaders containing random data in the correct Fashion ID range.

    B2B DIFFERENCE:
    item IDs are in [ITEM_ID_START..ITEM_ID_END], padding=182983.
    """

    # Task A: triples in unified ID space [0..182982]
    h = torch.randint(
        ITEM_ID_START,
        ITEM_ID_END + 1,
        (n_triples,)
    )

    r = torch.randint(
        0,
        3,
        (n_triples,)
    )

    t = torch.randint(
        ITEM_ID_START,
        ITEM_ID_END + 1,
        (n_triples,)
    )

    loader_a = DataLoader(
        TensorDataset(h, r, t),
        batch_size=batch_size,
        shuffle=True
    )

    # Task B: sequences in unified ID space, padding=182983
    item_seq = torch.full(
        (n_seqs, seq_len),
        182_983,
        dtype=torch.long
    )

    for i in range(n_seqs):

        sl = torch.randint(
            2,
            seq_len,
            (1,)
        ).item()

        item_seq[i, :sl] = torch.randint(
            ITEM_ID_START,
            ITEM_ID_END + 1,
            (sl,)
        )

    item_seq_len = (
        item_seq != 182_983
    ).sum(dim=1)

    pos_items = torch.randint(
        ITEM_ID_START,
        ITEM_ID_END + 1,
        (n_seqs,)
    )

    loader_b = DataLoader(
        TensorDataset(
            item_seq,
            item_seq_len,
            pos_items
        ),
        batch_size=batch_size,
        shuffle=True
    )

    return loader_a, loader_b


if __name__ == "__main__":

    print("=" * 70)
    print("ALTERNATE LOOP FASHION — SMOKE TEST")
    print("=" * 70)

    BATCH  = 32
    EPOCHS = 2
    NUM_ENTITIES = 182_983

    loader_a, loader_b = _make_fake_loaders_fashion(
        batch_size=BATCH
    )

    for mode in [
        'one_to_one',
        'epoch',
        'adaptive'
    ]:

        print(
            f"\n--- Test scheduling='{mode}' ---"
        )

        model = JointAlternateModelFashion(
            num_entities=182_984,
            num_relations=3,
            kge_dim=64,
            unified_padding_idx=182_983,
        )

        # Save initial weights to verify that they change
        w0 = (
            model
            .shared_embedding
            .embedding
            .weight
            .data
            .clone()
        )

        trainer = AlternateTrainerFashion(
            model=model,
            loader_a=loader_a,
            loader_b=loader_b,
            num_entities=NUM_ENTITIES,
            learning_rate=1e-2,
            scheduling=mode,
            device='cpu',
            log_every=999999,
        )

        history = trainer.train(
            num_epochs=EPOCHS
        )

        # ---- Check 1: correct number of epochs ----
        assert len(history) == EPOCHS, \
            f"Expected {EPOCHS} epochs, got {len(history)}"

        print(
            f"  Epochs: {len(history)} ✓"
        )

        # ---- Check 2: losses exist and are finite ----
        for ep in history:

            assert ep['batches_a'] > 0, \
                "No Task A batches!"

            assert ep['batches_b'] > 0, \
                "No Task B batches!"

            assert not torch.isnan(
                torch.tensor(
                    ep['loss_a_mean']
                )
            ), \
                f"loss_a is NaN at epoch {ep['epoch']}"

            assert not torch.isnan(
                torch.tensor(
                    ep['loss_b_mean']
                )
            ), \
                f"loss_b is NaN at epoch {ep['epoch']}"

        print(
            f"  Loss A: "
            f"{history[0]['loss_a_mean']:.4f} "
            f"→ {history[-1]['loss_a_mean']:.4f} ✓"
        )

        print(
            f"  Loss B: "
            f"{history[0]['loss_b_mean']:.4f} "
            f"→ {history[-1]['loss_b_mean']:.4f} ✓"
        )

        print(
            f"  Batches/epoch: "
            f"A={history[0]['batches_a']}, "
            f"B={history[0]['batches_b']}"
        )

        # ---- Check 3: weights changed ----
        w1 = (
            model
            .shared_embedding
            .embedding
            .weight
            .data
        )

        weight_diff = (
            w1 - w0
        ).abs().sum().item()

        assert weight_diff > 0, \
            "SharedEmbedding did not change after training!"

        print(
            f"  SharedEmbedding changed: "
            f"Δ={weight_diff:.4f} ✓"
        )

        # ---- Check 4: padding received no updates ----
        pad_norm = (
            model
            .shared_embedding
            .embedding
            .weight
            .data[182_983]
            .abs()
            .sum()
            .item()
        )

        assert pad_norm == 0.0, \
            f"Padding row received an update! norm={pad_norm:.6f}"

        print(
            f"  Padding (ID=182983) unchanged: "
            f"norm={pad_norm:.6f} ✓"
        )

        # ---- Check 5: gradients from both tasks ----
        # Run one separate Task A and Task B batch
        # and verify gradients on SharedEmbedding.

        model.zero_grad()

        h = torch.randint(
            ITEM_ID_START,
            ITEM_ID_END + 1,
            (8,)
        )

        r = torch.zeros(
            8,
            dtype=torch.long
        )

        t = torch.randint(
            ITEM_ID_START,
            ITEM_ID_END + 1,
            (8,)
        )

        loss_a = TransELoss(
            margin=8.684
        )(
            model.forward_kge(
                h,
                r,
                t
            ),
            model.forward_kge(
                h,
                r,
                torch.randint(
                    ITEM_ID_START,
                    ITEM_ID_END + 1,
                    (8,)
                )
            )
        )

        loss_a.backward()

        grad_a = (
            model
            .shared_embedding
            .embedding
            .weight
            .grad
            .abs()
            .sum()
            .item()
        )

        assert grad_a > 0, \
            "No gradient from Task A to SharedEmbedding!"

        model.zero_grad()

        seq = torch.full(
            (4, 10),
            182_983,
            dtype=torch.long
        )

        seq[:, :3] = torch.randint(
            ITEM_ID_START,
            ITEM_ID_END + 1,
            (4, 3)
        )

        slen = torch.tensor([
            3,
            3,
            3,
            3
        ])

        pos = torch.randint(
            ITEM_ID_START,
            ITEM_ID_END + 1,
            (4,)
        )

        urepr = model.forward_sasrec(
            seq,
            slen
        )

        pos_e = model.shared_embedding(
            pos
        )

        neg = torch.randint(
            ITEM_ID_START,
            ITEM_ID_END + 1,
            (4,)
        )

        neg_e = model.shared_embedding(
            neg
        )

        loss_b = SASRecBPRLoss()(
            (
                urepr * pos_e
            ).sum(-1).unsqueeze(1),

            (
                urepr.unsqueeze(1)
                * neg_e
            ).sum(-1)
        )

        loss_b.backward()

        grad_b = (
            model
            .shared_embedding
            .embedding
            .weight
            .grad
            .abs()
            .sum()
            .item()
        )

        assert grad_b > 0, \
            "No gradient from Task B to SharedEmbedding!"

        print(
            f"  Grad Task A={grad_a:.4f}, "
            f"Task B={grad_b:.4f} ✓"
        )

        # ---- Check 6: negative items are in the correct range ----
        pos_test = torch.randint(
            ITEM_ID_START,
            ITEM_ID_END + 1,
            (64,)
        )

        neg_test = trainer.sampler_b.sample(
            pos_test,
            num_neg=1
        )

        assert neg_test.min().item() >= ITEM_ID_START, \
            f"Negative < ITEM_ID_START: {neg_test.min().item()}"

        assert neg_test.max().item() <= ITEM_ID_END, \
            f"Negative > ITEM_ID_END: {neg_test.max().item()}"

        assert (neg_test == 182_983).sum() == 0, \
            "Padding sampled as a negative!"

        assert (neg_test == pos_test).sum() == 0, \
            "Negative == positive collision!"

        print(
            f"  Negatives in range "
            f"[{ITEM_ID_START}..{ITEM_ID_END}], "
            f"no padding, no collisions ✓"
        )

        print(
            f"  PASS: scheduling='{mode}'"
        )

    print("\n" + "=" * 70)
    print("ALTERNATE LOOP FASHION — SMOKE TEST PASSED")
    print("=" * 70)
    print()
    print("Summary:")
    print(
        f"  Supported scheduling: "
        f"one_to_one (default), epoch, adaptive"
    )
    print(
        f"  Task A: unified IDs "
        f"[0..{NUM_KG_ENT-1}], margin={8.684}"
    )
    print(
        f"  Task B: unified IDs "
        f"[{ITEM_ID_START}..{ITEM_ID_END}], BPR loss"
    )
    print(
        f"  Padding ({182_983}): "
        f"never updated during training"
    )