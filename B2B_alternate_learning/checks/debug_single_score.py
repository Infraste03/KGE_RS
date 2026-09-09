import os, sys, torch
sys.path.insert(0, "B2B_alternate_learning/")
sys.path.insert(0, "B2B_alternate_learning/models")

from load_kg import load_kg
from shared_embedding import SharedEmbedding
from task_a_transe import TaskATransE
from pykeen.triples import TriplesFactory
from pykeen.models import TransE
import pandas as pd

# 1. Build our model
kg = load_kg("data/processed/kg_train.tsv",
             "data/processed/taskA_valid.tsv",
             "data/processed/taskA_test.tsv", verbose=False)
shared = SharedEmbedding(num_entities=kg['num_entities'], embedding_dim=400)
task_a = TaskATransE(shared, num_relations=5, embedding_dim=400, normalize_entities=False)

sd = torch.load("results/step2_hpo/TransE/best_model.pt", weights_only=False)

shared.load_from_pykeen_transe(sd['entity_representations.0._embeddings.weight'],
                                kg['entity_to_id'], kg['entity_to_id'], verbose=False)
task_a.load_relation_weights_from_pykeen(sd['relation_representations.0._embeddings.weight'],
                                          kg['relation_to_id'], kg['relation_to_id'], verbose=False)

# 2. Build PyKEEN model
train_df = pd.read_csv("data/processed/kg_train.tsv", sep='\t')
factory = TriplesFactory.from_labeled_triples(train_df[['head','relation','tail']].values)
pk_model = TransE(triples_factory=factory, embedding_dim=400)
pk_model.load_state_dict(sd)
pk_model.eval()

# 3. Pick the FIRST test triple
test_triple = kg['test_triples'][0]
h, r, t = int(test_triple[0]), int(test_triple[1]), int(test_triple[2])
print(f"Triple: h={h}, r={r}, t={t}")

# Add this BEFORE the "OUR score at..." print
print(f"\n--- Diagnostic ---")
print(f"OUR shared norms: mean={shared.embedding.weight.norm(dim=1).mean().item():.4f}")
print(f"OUR relation norms: mean={task_a.relation_embedding.weight.norm(dim=1).mean().item():.4f}")
print(f"PYKEEN entity norms: mean={pk_model.entity_representations[0]._embeddings.weight.norm(dim=1).mean().item():.4f}")
print(f"PYKEEN relation norms: mean={pk_model.relation_representations[0]._embeddings.weight.norm(dim=1).mean().item():.4f}")
print()

h_vec = shared.embedding.weight[h]
r_vec = task_a.relation_embedding.weight[r]
t_vec = shared.embedding.weight[t]

diff = h_vec + r_vec - t_vec

# Print various forms of the norm to understand what PyKEEN uses
print(f"Manual ||h+r-t||_2 (Norma Euclidea vera) = {diff.norm(p=2).item():.4f}")
print(f"Manual ||h+r-t||_2_squared (Quadrato)  = {(diff**2).sum().item():.4f}")
print(f"Manual ||h+r-t||_1 (Manhattan / Somma assoluta) = {diff.norm(p=1).item():.4f}")
print()

# 4. Compute score with OUR model
with torch.no_grad():
    our_scores = task_a.score_all_tails(torch.LongTensor([h]), torch.LongTensor([r])).squeeze(0)
    print(f"OUR score at t={t}: {our_scores[t].item():.6f}")
    print(f"OUR top-5 scores: {our_scores.topk(5).values.tolist()}")
    print(f"OUR top-5 indices: {our_scores.topk(5).indices.tolist()}")

# 5. Compute score with PyKEEN model
with torch.no_grad():
    pk_scores = pk_model.score_t(torch.tensor([[h, r]])).squeeze(0)
    print(f"PYKEEN score at t={t}: {pk_scores[t].item():.6f}")
    print(f"PYKEEN top-5 scores: {pk_scores.topk(5).values.tolist()}")
    print(f"PYKEEN top-5 indices: {pk_scores.topk(5).indices.tolist()}")