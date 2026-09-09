"""
verify_full_transe.py - Verify that best_model.pt (TransE full) gives the
expected metrics, recalculating MRR/Hits@10 on the full test set.

Uso:
    python verify_full_transe.py
"""

import sys
import os
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, os.path.join(THIS_DIR, "models"))

from load_kg import load_kg
from task_a_transe import TaskATransE
from shared_embedding import SharedEmbedding
from eval_utils import evaluate_task_a

PROJECT_ROOT = os.path.dirname(THIS_DIR)

BEST_MODEL_PATH = os.path.join(
    PROJECT_ROOT, "results", "step2_hpo", "TransE", "best_model.pt"
)
KG_TRAIN_PATH = os.path.join(
    PROJECT_ROOT, "data", "processed", "kg_train.tsv"
)
KG_VALID_PATH = os.path.join(
    PROJECT_ROOT, "data", "processed", "taskA_valid.tsv"
)
KG_TEST_PATH = os.path.join(
    PROJECT_ROOT, "data", "processed", "taskA_test.tsv"
)

EMBEDDING_DIM = 400

def main():
    print("=" * 70)
    print("CHECK best_model.pt (TransE FULL) - MRR/Hits@10 recalculation")
    print("=" * 70)

    kg = load_kg(KG_TRAIN_PATH, KG_VALID_PATH, KG_TEST_PATH, verbose=False)
    print(f"num_entities={kg['num_entities']}, num_relations={kg['num_relations']}")
    kg['test_triples'] = torch.from_numpy(kg['test_triples']).long()

    shared = SharedEmbedding(num_entities=kg['num_entities'], embedding_dim=EMBEDDING_DIM, padding_idx=None)
    task_a = TaskATransE(shared_embedding=shared, num_relations=kg['num_relations'],
                          embedding_dim=EMBEDDING_DIM, p_norm=1, normalize_entities=False)

    state_dict = torch.load(BEST_MODEL_PATH, map_location="cpu", weights_only=False)
    entity_w = state_dict["entity_representations.0._embeddings.weight"]
    relation_w = state_dict["relation_representations.0._embeddings.weight"]

    with torch.no_grad():
        shared.embedding.weight[:] = entity_w
        task_a.relation_embedding.weight[:] = relation_w

    type_index = {"item": torch.LongTensor(kg["entity_type_indices"]["item"])}

    class DummyModel:
        pass
    dummy = DummyModel()
    dummy.task_a = task_a
    dummy.eval = lambda: None
    dummy.parameters = lambda: iter([shared.embedding.weight])

    metrics = evaluate_task_a(dummy, kg, type_index, relation_label="compatible_with")

    print("\nRESULTS:")
    print(f"  MRR      : {metrics['MRR']:.4f}")
    print(f"  Hits@10  : {metrics['Hits@10']:.4f}")
    print("\nCompare these values with those you had recorded for the full model")
    print("(from your original Step 2 / from the RecSys 2026 paper submitted).")


if __name__ == "__main__":
    main()