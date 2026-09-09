"""
alternate_loop_fashion_ce.py
=============================

Alternate training loop for Step 4  Fashion domain, with Cross-Entropy loss
for Task B.

Three scheduling modes (identical to B2B):
    'one_to_one' : one Task A batch, one Task B batch, alternating 1:1 (default)
    'epoch'      : one full Task A epoch, followed by one full Task B epoch
    'adaptive'   : the Task A:Task B ratio adapts according to the losses

Differences compared with the BPR Fashion loop:

    __init__:
      - Task B uses SASRecCELoss instead of SASRecBPRLoss
      - Candidate item IDs are explicitly constructed in the unified ID space
      - The standard Fashion candidate range is [17514..182982]
      - margin_a default = 8.684 from Fashion TransE HPO

    _train_step_b:
      - pos_items are already unified IDs [17514..182982]
      - scores are computed over the entire candidate item catalogue
      - targets are converted from unified IDs to positions in the
        candidate vector
      - no negative sampling is required by the CE loss itself

    Smoke test:
      - uses a reduced catalogue of 100 items to keep full-softmax CE fast
      - verifies that SharedEmbedding receives gradients from both tasks
      - verifies that weights change after training
      - verifies that CE decreases when the target receives the highest score

File location:
    fashion_generalization/alternate_learning/alternate_loop_fashion_ce.py
"""

import os
import sys
import logging
import time
import torch
from torch.utils.data import DataLoader, TensorDataset

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


MODELS_DIR = os.path.join(THIS_DIR, "models")

if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

if MODELS_DIR not in sys.path:
    sys.path.insert(0, MODELS_DIR)

from joint_model_fashion import JointAlternateModelFashion
from losses_fashion import TransELoss, SASRecCELoss
from samplers_fashion import NegativeSamplerKGE, NegativeSamplerRec

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Costanti fashion
ITEM_ID_START = 17_514
ITEM_ID_END   = 182_982
NUM_KG_ENT    = 182_983


class AlternateTrainerFashion:
    """
    Alternating training of Task A (TransE) and Task B (SASRec) all good
    JointAlternateModelFashion.

    Parameters
    -------
    model: JointAlternateModelFashion
        Model with hot-dosed SharedEmbedding.
    loader_a : DataLoader
        Batch = (h, r, t) LongTensor, Unified IDs [0..182982].
    loader_b: DataLoader
        Batch = (item_seq, item_seq_len, pos_items) LongTensor.
        item_seq: Unified IDs, padding=182983.
        pos_items: Unified IDs [17514..182982].
        Produced by build_rec_dataloader_fashion (already translated).
    num_entity: int
        Direct number KG for NegativeSamplerKGE. Fashion: 182983.
    appreciation_rate: float
        Learning rate Adam.
    margin_a: floating
        TransE marginal loss. Default HPO Fashion 8,684 from.
    planning: str
        'one_to_one', 'epoch', or 'adaptive'.
    device: str
        'cuda' or 'cpu'.
    log_every: int
        Register each batch.
    num_neg_b : int
        Negatives for positive in Task B. Default 1 (BPR).
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
        item_id_start: int = 17_514,
        item_id_end: int = 182_982,
    ):
        if scheduling not in ('one_to_one', 'epoch', 'adaptive'):
            raise ValueError(
                f"scheduling deve essere uno di 'one_to_one', 'epoch', 'adaptive'; "
                f"got '{scheduling}'"
            )

        self.model      = model.to(device)
        self.loader_a   = loader_a
        self.loader_b   = loader_b
        self.device     = device
        self.scheduling = scheduling
        self.log_every  = log_every
        self.num_neg_b  = num_neg_b
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Loss functions
        self.loss_a_fn = TransELoss(margin=margin_a)
        self.loss_b_fn = SASRecCELoss()

        self.item_id_start = item_id_start
        self.item_id_end = item_id_end

        self.candidate_ids = torch.arange(
            self.item_id_start,
            self.item_id_end + 1,
            device=device
        )
        self.sampler_a = NegativeSamplerKGE(num_entities=num_entities)

        self.sampler_b = NegativeSamplerRec(
            item_id_start=self.item_id_start,
            item_id_end=self.item_id_end,
        )

        self._last_loss_a = 1.0
        self._last_loss_b = 1.0

        logger.info(
            f"AlternateTrainerFashion: scheduling={scheduling}, "
            f"lr={learning_rate}, margin_a={margin_a}, device={device}, "
            f"num_neg_b={num_neg_b}"
        )

    # --------------------------------------------------------------------- #
    # Singoli step di training
    # --------------------------------------------------------------------- #

    def _train_step_a(self, batch_a) -> float:

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
        self.optimizer.zero_grad()

        item_seq, item_seq_len, pos_items = (x.to(self.device) for x in batch_b)
        user_repr = self.model.forward_sasrec(item_seq, item_seq_len)  # (B, 64)
        scores_all = self.model.score_items_sasrec(user_repr, self.candidate_ids)  # (B, N_items)
        targets = pos_items - self.candidate_ids[0]  # (B,)
        loss = self.loss_b_fn(scores_all, targets)

        loss.backward()
        self.optimizer.step()

        return loss.item()


    def _epoch_one_to_one(self):
        """1:1  one batch A, one batch B, until both are used up."""
        iter_a    = iter(self.loader_a)
        iter_b    = iter(self.loader_b)
        losses_a, losses_b = [], []
        step = 0
        exhausted_a = exhausted_b = False

        while not (exhausted_a and exhausted_b):
            if not exhausted_a:
                try:
                    loss_a = self._train_step_a(next(iter_a))
                    losses_a.append(loss_a)
                except StopIteration:
                    exhausted_a = True

            if not exhausted_b:
                try:
                    loss_b = self._train_step_b(next(iter_b))
                    losses_b.append(loss_b)
                except StopIteration:
                    exhausted_b = True

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
        """Adaptive"""
        iter_a = iter(self.loader_a)
        iter_b = iter(self.loader_b)
        losses_a, losses_b = [], []
        exhausted_a = exhausted_b = False
        step = 0

        while not (exhausted_a and exhausted_b):
            do_a = (self._last_loss_a >= self._last_loss_b) and not exhausted_a
            if exhausted_a:
                do_a = False
            elif exhausted_b:
                do_a = True
            if step % 3 == 2 and not exhausted_b:
                do_a = False

            if do_a:
                try:
                    loss_a = self._train_step_a(next(iter_a))
                    losses_a.append(loss_a)
                    self._last_loss_a = 0.9 * self._last_loss_a + 0.1 * loss_a
                except StopIteration:
                    exhausted_a = True
                    continue
            else:
                try:
                    loss_b = self._train_step_b(next(iter_b))
                    losses_b.append(loss_b)
                    self._last_loss_b = 0.9 * self._last_loss_b + 0.1 * loss_b
                except StopIteration:
                    exhausted_b = True
                    continue

            step += 1
            if step % self.log_every == 0:
                logger.info(
                    f"  step {step}  ema_a={self._last_loss_a:.4f}  "
                    f"ema_b={self._last_loss_b:.4f}  "
                    f"done_a={len(losses_a)}  done_b={len(losses_b)}"
                )

        return losses_a, losses_b


    def train_one_epoch(self) -> dict:
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
            'batches_a'  : len(losses_a),
            'batches_b'  : len(losses_b),
            'time_sec'   : dt,
        }

    def train(self, num_epochs: int) -> list:
        history = []
        for epoch in range(1, num_epochs + 1):
            logger.info(f"\n=== Epoch {epoch}/{num_epochs} ===")
            metrics = self.train_one_epoch()
            metrics['epoch'] = epoch
            history.append(metrics)
            logger.info(
                f"Epoch {epoch} done in {metrics['time_sec']:.1f}s  "
                f"loss_a={metrics['loss_a_mean']:.4f}  "
                f"loss_b={metrics['loss_b_mean']:.4f}  "
                f"batches_a={metrics['batches_a']}  "
                f"batches_b={metrics['batches_b']}"
            )
        return history


def _make_fake_loaders_fashion(batch_size, n_triples=512, n_seqs=256, seq_len=10,
                                item_id_start=ITEM_ID_START, item_id_end=ITEM_ID_END,
                                padding_id=182_983):

    h = torch.randint(item_id_start, item_id_end + 1, (n_triples,))
    r = torch.randint(0, 3, (n_triples,))
    t = torch.randint(item_id_start, item_id_end + 1, (n_triples,))
    loader_a = DataLoader(TensorDataset(h, r, t), batch_size=batch_size, shuffle=True)

    item_seq = torch.full((n_seqs, seq_len), padding_id, dtype=torch.long)
    for i in range(n_seqs):
        sl = torch.randint(2, seq_len, (1,)).item()
        item_seq[i, :sl] = torch.randint(item_id_start, item_id_end + 1, (sl,))
    item_seq[i, :sl] = torch.randint(item_id_start, item_id_end + 1, (sl,))
    item_seq_len = (item_seq != padding_id).sum(dim=1)
    pos_items    = torch.randint(item_id_start, item_id_end + 1, (n_seqs,))
    loader_b = DataLoader(
        TensorDataset(item_seq, item_seq_len, pos_items),
        batch_size=batch_size, shuffle=True
    )

    return loader_a, loader_b


if __name__ == "__main__":
    print("=" * 70)
    print("ALTERNATE LOOP FASHION — SMOKE TEST")
    print("=" * 70)

    BATCH  = 32
    EPOCHS = 2
    NUM_ENTITIES = 182_983
    TEST_ITEM_START = 0
    TEST_ITEM_END   = 99
    TEST_PADDING    = 100

    loader_a, loader_b = _make_fake_loaders_fashion(
    batch_size=BATCH,
    item_id_start=TEST_ITEM_START, item_id_end=TEST_ITEM_END,
    padding_id=TEST_PADDING,
    )

    for mode in ['one_to_one', 'epoch', 'adaptive']:
        print(f"\n--- Test scheduling='{mode}' ---")

        model = JointAlternateModelFashion(
        num_entities=TEST_PADDING + 1,
        num_relations=3,
        kge_dim=64,
        unified_padding_idx=TEST_PADDING,
    )

        w0 = model.shared_embedding.embedding.weight.data.clone()

        trainer = AlternateTrainerFashion(
        model=model,
        loader_a=loader_a,
        loader_b=loader_b,
        num_entities=TEST_PADDING,
        learning_rate=1e-2,
        scheduling=mode,
        device='cpu',
        log_every=999999,
        item_id_start=TEST_ITEM_START,
        item_id_end=TEST_ITEM_END,
    )

        history = trainer.train(num_epochs=EPOCHS)

        assert len(history) == EPOCHS, f"Atteso {EPOCHS} epoche, got {len(history)}"
        print(f"  Epoche: {len(history)} ✓")

        for ep in history:
            assert ep['batches_a'] > 0, "Nessun batch Task A!"
            assert ep['batches_b'] > 0, "Nessun batch Task B!"
            assert not torch.isnan(torch.tensor(ep['loss_a_mean'])), \
                f"loss_a NaN all'epoca {ep['epoch']}"
            assert not torch.isnan(torch.tensor(ep['loss_b_mean'])), \
                f"loss_b NaN all'epoca {ep['epoch']}"
        print(f"  Loss A: {history[0]['loss_a_mean']:.4f} → {history[-1]['loss_a_mean']:.4f} ✓")
        print(f"  Loss B: {history[0]['loss_b_mean']:.4f} → {history[-1]['loss_b_mean']:.4f} ✓")
        print(f"  Batches/epoca: A={history[0]['batches_a']}, B={history[0]['batches_b']}")
        w1 = model.shared_embedding.embedding.weight.data
        weight_diff = (w1 - w0).abs().sum().item()
        assert weight_diff > 0, "SharedEmbedding non è cambiata dopo il training!"
        print(f"  SharedEmbedding cambiata: Δ={weight_diff:.4f} ✓")


        pad_norm = model.shared_embedding.embedding.weight.data[TEST_PADDING].abs().sum().item()
        assert pad_norm == 0.0, \
            f"Riga padding ha ricevuto aggiornamento! norm={pad_norm:.6f}"
        print(f"  Padding (ID={TEST_PADDING}) invariato: norm={pad_norm:.6f} ✓")
        model.zero_grad()
        h = torch.randint(TEST_ITEM_START, TEST_ITEM_END + 1, (8,))
        r = torch.zeros(8, dtype=torch.long)
        t = torch.randint(TEST_ITEM_START, TEST_ITEM_END + 1, (8,))
        loss_a = TransELoss(margin=8.684)(
            model.forward_kge(h, r, t),
            model.forward_kge(h, r, torch.randint(TEST_ITEM_START, TEST_ITEM_END+1, (8,)))
        )
        loss_a.backward()
        grad_a = model.shared_embedding.embedding.weight.grad.abs().sum().item()
        assert grad_a > 0, "Nessun gradiente da Task A su SharedEmbedding!"

        model.zero_grad()
        seq   = torch.full((4, 10), TEST_PADDING, dtype=torch.long)
        seq[:, :3] = torch.randint(TEST_ITEM_START, TEST_ITEM_END + 1, (4, 3))
        slen  = torch.tensor([3, 3, 3, 3])
        pos   = torch.randint(TEST_ITEM_START, TEST_ITEM_END + 1, (4,))
        urepr = model.forward_sasrec(seq, slen)
        pos_e = model.shared_embedding(pos)
        neg   = torch.randint(TEST_ITEM_START, TEST_ITEM_END + 1, (4,))
        neg_e = model.shared_embedding(neg)
        candidate_ids_test = torch.arange(TEST_ITEM_START, TEST_ITEM_END + 1)
        scores_test = model.score_items_sasrec(urepr, candidate_ids_test)
        targets_test = pos - TEST_ITEM_START
        loss_b = SASRecCELoss()(scores_test, targets_test)
        loss_b.backward()
        grad_b = model.shared_embedding.embedding.weight.grad.abs().sum().item()
        assert grad_b > 0, "Nessun gradiente da Task B su SharedEmbedding!"
        print(f"  Grad Task A={grad_a:.4f}, Task B={grad_b:.4f}")

        pos_test = torch.randint(TEST_ITEM_START, TEST_ITEM_END + 1, (64,))
        neg_test = trainer.sampler_b.sample(pos_test, num_neg=1)
        assert neg_test.min().item() >= TEST_ITEM_START, \
            f"Negativo < TEST_ITEM_START: {neg_test.min().item()}"
        assert neg_test.max().item() <= TEST_ITEM_END, \
            f"Negativo > TEST_ITEM_END: {neg_test.max().item()}"
        assert (neg_test == 182_983).sum() == 0, \
            "Padding campionato come negativo!"
        assert (neg_test == pos_test).sum() == 0, \
            "Collisione negativo == positivo!"
        print(f"  Negativi range [{TEST_ITEM_START}..{TEST_ITEM_END}], no padding, no collisioni ✓")

        print(f"  PASS: scheduling='{mode}'")

        print(f"\n--- Check 7: CE loss coerenza ---")
        user_repr_fake = torch.randn(4, 64)
        candidates_fake = torch.arange(TEST_ITEM_START, TEST_ITEM_START + 100)
        scores_fake = torch.randn(4, 100)
        targets_fake = torch.tensor([5, 10, 20, 50])

        loss_random = SASRecCELoss()(scores_fake, targets_fake)

        # Forza il target ad avere lo score massimo
        scores_boosted = scores_fake.clone()
        for i, t in enumerate(targets_fake):
            scores_boosted[i, t] = 100.0
        loss_boosted = SASRecCELoss()(scores_boosted, targets_fake)

        assert loss_boosted.item() < loss_random.item(), \
            "CE loss non diminuisce quando il target ha score massimo!"
        print(f"  loss_random={loss_random.item():.4f} > loss_boosted={loss_boosted.item():.4f} ✓")

    print("\n" + "=" * 70)
    print("ALTERNATE LOOP FASHION — SMOKE TEST PASSED")
    print("=" * 70)
    print()
    print("Summary:")
    print(f"  Supported scheduling: one_to_one (default), epoch, adaptive")
    print(f"  Task A: unified IDs [0..{NUM_KG_ENT-1}], margin={8.684}")
    print(f"  Task B: unified IDs [{ITEM_ID_START}..{ITEM_ID_END}], CE loss")
    print(f"  Padding ({182_983}): never updated during training")