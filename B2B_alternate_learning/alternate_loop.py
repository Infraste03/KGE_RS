"""
alternate_loop.py — Alternating training cycle for Phase 4.

Three scheduling modes (selectable from config / CLI):

    'one_to_one': one batch Task A, one batch Task B, alternating 1:1
                    (default; the one we will use in the main experiment)
    'epoch': an integer epoch of Task A, then an integer epoch of Task B
                    (ablation: simpler, but SharedEmbedding receives gradients
                     from a single task for long periods)
    'adaptive': Task A:Task B ratio adapts based on losses
                    (ablation: the task with the highest loss receives the most batches)

What this file does NOT do:
    - Doesn't load real data (run_step4.py does)
    - Does not warm start from Step 2/3 (run_step4.py does)
    - Does not save checkpoint (run_step4.py does)
    - Does not calculate evaluation metrics (run_step4.py does)

What does it do:
    - Defines the AlternateTrainer class
    - Implement the three scheduling modes
    - Exposes metrics for each epoch (loss_a_mean, loss_b_mean, battles_a, battles_b)
    - Smoke test integrated with fake data to validate the mechanics
"""

import os
import sys
import logging
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(THIS_DIR, "models")
if MODELS_DIR not in sys.path:
    sys.path.insert(0, MODELS_DIR)

from joint_model import JointAlternateModel
from losses import TransELoss, SASRecBPRLoss
from samplers import NegativeSamplerKGE, NegativeSamplerRec

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# =========================================================================== #
# AlternateTrainer
# =========================================================================== #

class AlternateTrainer:
    """
    Alternating training of Task A (TransE) and Task B (SASRec) on the same
    Joint alternative model.

    Parameters
    ----------
    model : JointAlternateModel
        The model with SharedEmbedding, TaskATransE, TaskBSASRec already initialized
        (possibly warm-started from Step 2/3).
    loader_a : DataLoader
        DataLoader for Task A. Each batch is one tuple (h, r, t) of tensors long.
    loader_b : DataLoader
        DataLoader for Task B. Each batch is one tuple (item_seq, item_seq_len, pos_items)
        of tensors long. item_seq has shape (B, T), pos_items shape (B,).
    num_entities : int
        For the negative sampler of Task A.
    num_items : int
        For the negative sampler of Task B (= dataset.item_num of RecBole).
    learning_rate : float
        Learning rate for Adam (default 1e-3).
    margin_a : float
        TransE Loss Margin (default 5.19 from HPO Phase 2).
    scheduling : str
        One between {'one_to_one', 'epoch', 'adaptive'}.
    device : str
        'cuda' or 'cpu'.
    log_every : int
        Print the loss every log_every batch.
    """

    def __init__(
        self,
        model: JointAlternateModel,
        loader_a: DataLoader,
        loader_b: DataLoader,
        num_entities: int,
        num_items: int,
        learning_rate: float = 1e-3,
        margin_a: float = 5.19,
        scheduling: str = 'one_to_one',
        device: str = 'cpu',
        log_every: int = 100,
        num_neg_b: int = 1,
    ):
        if scheduling not in ('one_to_one', 'epoch', 'adaptive'):
            raise ValueError(
                f"scheduling must be one of 'one_to_one', 'epoch', 'adaptive'; got {scheduling}"
            )

        self.model = model.to(device)
        self.loader_a = loader_a
        self.loader_b = loader_b
        self.device = device
        self.scheduling = scheduling
        self.log_every = log_every
        self.num_neg_b = num_neg_b
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        # Loss
        self.loss_a_fn = TransELoss(margin=margin_a)
        self.loss_b_fn = SASRecBPRLoss()
        self.sampler_a = NegativeSamplerKGE(num_entities=num_entities)
        padding_idx = getattr(model.task_b, 'padding_idx', None)
        self.sampler_b = NegativeSamplerRec(num_items=num_items, padding_idx=padding_idx)
        # Loss memory for adaptive mode
        self._last_loss_a = 1.0
        self._last_loss_b = 1.0

        logger.info(
            f"AlternateTrainer initialized: scheduling={scheduling}, lr={learning_rate}, "
            f"margin_a={margin_a}, device={device}"
        )

    # --------------------------------------------------------------------- #
    # Single training steps (one batch)
    # --------------------------------------------------------------------- #

    def _train_step_a(self, batch_a):
        """A single batch of Task A. The loss returns as a float."""
        h, r, t = (x.to(self.device) for x in batch_a)
        h_neg, r_neg, t_neg = self.sampler_a.sample(h, r, t)
        pos_scores = self.model.forward_kge(h, r, t)
        neg_scores = self.model.forward_kge(h_neg, r_neg, t_neg)
        loss = self.loss_a_fn(pos_scores, neg_scores)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _train_step_b(self, batch_b):
        self.optimizer.zero_grad()
        item_seq, item_seq_len, pos_items = (x.to(self.device) for x in batch_b)
        user_repr = self.model.forward_sasrec(item_seq, item_seq_len)
        pos_emb = self.model.shared_embedding(pos_items)
        neg_items = self.sampler_b.sample(pos_items, num_neg=self.num_neg_b)
        neg_items = neg_items.to(self.device)
        neg_emb = self.model.shared_embedding(neg_items)
        # BPR score and loss calculation
        pos_score = (user_repr * pos_emb).sum(dim=-1)  # (batch)
        # user_repr: (batch, 400) -> (batch, 1, 400)
        # neg_emb: (batch, num_neg, 400)
        neg_score = (user_repr.unsqueeze(1) * neg_emb).sum(dim=-1)  # (batch, num_neg)
        loss_b = self.loss_b_fn(pos_score.unsqueeze(1), neg_score)

        loss_b.backward()
        self.optimizer.step()

        return loss_b.item()


    # --------------------------------------------------------------------- #
    # Loop for a single epoch, in the three modes
    # --------------------------------------------------------------------- #

    def _epoch_one_to_one(self):
        """1:1 — a Task A batch, a Task B batch, until both are exhausted."""
        iter_a = iter(self.loader_a)
        iter_b = iter(self.loader_b)
        losses_a, losses_b = [], []
        step = 0

        while True:
            # Task A step
            try:
                batch_a = next(iter_a)
                loss_a = self._train_step_a(batch_a)
                losses_a.append(loss_a)
            except StopIteration:
                batch_a = None

            # Task B step
            try:
                batch_b = next(iter_b)
                loss_b = self._train_step_b(batch_b)
                losses_b.append(loss_b)
            except StopIteration:
                batch_b = None

            if batch_a is None and batch_b is None:
                break

            step += 1
            if step % self.log_every == 0:
                la = losses_a[-1] if losses_a else float('nan')
                lb = losses_b[-1] if losses_b else float('nan')
                logger.info(f"  step {step}  loss_a={la:.4f}  loss_b={lb:.4f}")

        return losses_a, losses_b

    def _epoch_epoch(self):
        """Full epoch of Task A, then full epoch of Task B."""
        losses_a, losses_b = [], []

        logger.info("  [Task A phase]")
        for i, batch_a in enumerate(self.loader_a):
            loss_a = self._train_step_a(batch_a)
            losses_a.append(loss_a)
            if (i + 1) % self.log_every == 0:
                logger.info(f"    A step {i+1}  loss_a={loss_a:.4f}")

        logger.info("  [Task B phase]")
        for i, batch_b in enumerate(self.loader_b):
            loss_b = self._train_step_b(batch_b)
            losses_b.append(loss_b)
            if (i + 1) % self.log_every == 0:
                logger.info(f"    B step {i+1}  loss_b={loss_b:.4f}")

        return losses_a, losses_b

    def _epoch_adaptive(self):
        """
        Adaptive — the ratio of Task A to Task B adapts based on the losses.
        Simple strategy: at each step, the task with the highest average loss is performed.
        When a loader is exhausted, the other is completed.
        """
        iter_a = iter(self.loader_a)
        iter_b = iter(self.loader_b)
        losses_a, losses_b = [], []
        exhausted_a, exhausted_b = False, False
        step = 0

        while not (exhausted_a and exhausted_b):
            do_a = (self._last_loss_a >= self._last_loss_b) and not exhausted_a
            if exhausted_a:
                do_a = False
            elif exhausted_b:
                do_a = True
            # Minimum constraint: at least 1 in 3 steps always goes to Task B
            if step % 3 == 2 and not exhausted_b:
                do_a = False

            if do_a:
                try:
                    batch_a = next(iter_a)
                    loss_a = self._train_step_a(batch_a)
                    losses_a.append(loss_a)
                    self._last_loss_a = 0.9 * self._last_loss_a + 0.1 * loss_a
                except StopIteration:
                    exhausted_a = True
                    continue
            else:
                try:
                    batch_b = next(iter_b)
                    loss_b = self._train_step_b(batch_b)
                    losses_b.append(loss_b)
                    self._last_loss_b = 0.9 * self._last_loss_b + 0.1 * loss_b
                except StopIteration:
                    exhausted_b = True
                    continue

            step += 1
            if step % self.log_every == 0:
                logger.info(
                    f"  step {step}  ema_a={self._last_loss_a:.4f}  "
                    f"ema_b={self._last_loss_b:.4f}  done_a={len(losses_a)}  done_b={len(losses_b)}"
                )

        return losses_a, losses_b

    # --------------------------------------------------------------------- #
    # public API
    # --------------------------------------------------------------------- #

    def train_one_epoch(self):
        """An epoch of training in the chosen mode. A metrics dict returns."""
        self.model.train()
        t0 = time.time()

        if self.scheduling == 'one_to_one':
            losses_a, losses_b = self._epoch_one_to_one()
        elif self.scheduling == 'epoch':
            losses_a, losses_b = self._epoch_epoch()
        elif self.scheduling == 'adaptive':
            losses_a, losses_b = self._epoch_adaptive()
        else:
            raise ValueError(f"Unknown scheduling: {self.scheduling}")

        dt = time.time() - t0
        return {
            'loss_a_mean': sum(losses_a) / max(len(losses_a), 1),
            'loss_b_mean': sum(losses_b) / max(len(losses_b), 1),
            'batches_a': len(losses_a),
            'batches_b': len(losses_b),
            'time_sec': dt,
        }

    def train(self, num_epochs: int):
        """Launch num_epochs epochs of training. Returns the list of metrics per epoch."""
        history = []
        for epoch in range(1, num_epochs + 1):
            logger.info(f"\n=== Epoch {epoch}/{num_epochs} ===")
            metrics = self.train_one_epoch()
            metrics['epoch'] = epoch
            history.append(metrics)
            logger.info(
                f"Epoch {epoch} done in {metrics['time_sec']:.1f}s  "
                f"loss_a_mean={metrics['loss_a_mean']:.4f}  "
                f"loss_b_mean={metrics['loss_b_mean']:.4f}  "
                f"batches_a={metrics['batches_a']}  batches_b={metrics['batches_b']}"
            )
        return history


# =========================================================================== #
# SMOKE TEST
# =========================================================================== #

def _make_fake_loaders(num_entities, num_relations, num_items, batch_size, max_seq_len):
    """Create two fake data loaders for smoke testing."""
    # Task A: 1024 random triples (h, r, t)
    h = torch.randint(0, num_entities, (1024,))
    r = torch.randint(0, num_relations, (1024,))
    t = torch.randint(0, num_entities, (1024,))
    ds_a = TensorDataset(h, r, t)
    loader_a = DataLoader(ds_a, batch_size=batch_size, shuffle=True)
    # Task B: 512 random sequences of items, with random lengths and positive items
    item_seq = torch.randint(1, num_items, (512, max_seq_len))
    item_seq_len = torch.randint(1, max_seq_len + 1, (512,))
    pos_items = torch.randint(1, num_items, (512,))
    ds_b = TensorDataset(item_seq, item_seq_len, pos_items)
    loader_b = DataLoader(ds_b, batch_size=batch_size, shuffle=True)

    return loader_a, loader_b


if __name__ == "__main__":
    print("=" * 70)
    print("ALTERNATE LOOP SMOKE TEST")
    print("=" * 70)

    # Small size for quick smoke test
    NUM_ENTITIES = 1000
    NUM_RELATIONS = 5
    NUM_ITEMS = 500
    BATCH = 32
    SEQ_LEN = 20
    EPOCHS = 2

    # Build fake model and dataloaders
    model = JointAlternateModel(
        num_entities=NUM_ENTITIES,
        num_relations=NUM_RELATIONS,
        num_items=NUM_ITEMS,
        kge_dim=32,
        sasrec_dim=16,
        max_seq_length=SEQ_LEN,
    )
    loader_a, loader_b = _make_fake_loaders(
        NUM_ENTITIES, NUM_RELATIONS, NUM_ITEMS, BATCH, SEQ_LEN
    )

    for mode in ['one_to_one', 'epoch', 'adaptive']:
        print(f"\n--- Test scheduling = '{mode}' ---")
        model = JointAlternateModel(
            num_entities=NUM_ENTITIES,
            num_relations=NUM_RELATIONS,
            num_items=NUM_ITEMS,
            kge_dim=32,
            sasrec_dim=16,
            max_seq_length=SEQ_LEN,
        )

        trainer = AlternateTrainer(
            model=model,
            loader_a=loader_a,
            loader_b=loader_b,
            num_entities=NUM_ENTITIES,
            num_items=NUM_ITEMS,
            learning_rate=1e-2,
            scheduling=mode,
            device='cpu',
            log_every=999999,
        )

        history = trainer.train(num_epochs=EPOCHS)
        assert len(history) == EPOCHS, f"Expected {EPOCHS} epochs"
        loss_a_start = history[0]['loss_a_mean']
        loss_a_end = history[-1]['loss_a_mean']
        loss_b_start = history[0]['loss_b_mean']
        loss_b_end = history[-1]['loss_b_mean']

        print(f"  Task A: loss start={loss_a_start:.4f}, end={loss_a_end:.4f}")
        print(f"  Task B: loss start={loss_b_start:.4f}, end={loss_b_end:.4f}")
        print(f"  Batches per epoch: A={history[0]['batches_a']}, B={history[0]['batches_b']}")
        print(f"  PASS: mode '{mode}' runs for {EPOCHS} epochs without errors")

    print("\n" + "=" * 70)
    print("ALTERNATE LOOP SMOKE TEST PASSED")
    print("=" * 70)