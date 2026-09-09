from scipy.stats import wilcoxon

hybrid_one_to_one = [0.2835, 0.2913, 0.2734, 0.27, 0.25]
sasrec = [0.1869, 0.193, 0.1837, 0.1733, 0.18]

stat, p = wilcoxon(hybrid_one_to_one, sasrec, alternative='greater')
print(f"p-value = {p:.4f}")