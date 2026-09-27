"""How many independent directions are behind the valence result? 7B vectors at layer 19. Run from repo root."""
import torch, numpy as np, pandas as pd
d = torch.load("data/steer-runpod/qwen2.5-coder-7b-instruct/vectors.pt", weights_only=False)
em = d["emotions"]; V = d["V"][:, 19].double().numpy()
U = V / np.linalg.norm(V, axis=1, keepdims=True)
C = pd.DataFrame(U @ U.T, index=em, columns=em).round(2)
print("cosine similarity, layer 19\n", C.to_string(), "\n")
print("sum of the 12 vectors / mean norm:", round(np.linalg.norm(V.sum(0)) / np.linalg.norm(V, axis=1).mean(), 3))
# valence axis = first principal component of the 12 vectors, oriented so calm is positive
_, S, Wt = np.linalg.svd(V - V.mean(0), full_matrices=False)
pc1 = Wt[0] * np.sign(U[em.index("calm")] @ Wt[0])
print("variance explained by PC1..3:", (S**2 / (S**2).sum())[:3].round(2))
proj = pd.Series(U @ pc1, index=em)
# every run steering direction: sign * vector, its cosine with the valence axis, and the behaviour
E = pd.read_parquet("data/write-up/labels/noexit/episodes_final.parquet")
E = E[(E["size"] == "7b") & (E.a == 0.5) & (E.grp == "emotion")]
r = E.groupby("cond")[["succ", "imp"]].mean().mul(100).round(0)
r["cos_with_valence"] = [(1 if c[0] == "+" else -1) * proj[c[1:]] for c in r.index]
r = r.sort_values("cos_with_valence")
print(r.to_string())
side = np.sign(r.cos_with_valence); dom = np.where(r.succ > r.imp, "false success", np.where(r.imp > r.succ, "not possible", "neither"))
print("\npositive side of axis:", pd.Series(dom[side > 0]).value_counts().to_dict(),
      "| negative side:", pd.Series(dom[side < 0]).value_counts().to_dict())
print("rank correlation (cos, false success):", round(r.cos_with_valence.rank().corr(r.succ.rank()), 2),
      "| (cos, not possible):", round(r.cos_with_valence.rank().corr(r.imp.rank()), 2))
