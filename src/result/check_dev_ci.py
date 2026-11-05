"""한 번 올린 대가 점검의 반복 단위 신뢰구간: 250개 상태는 시장 반복 10개에서 나와 같은 반복 안의 상태끼리 상관이 있다.
check_dev.json, check_dev_x2.json의 m별 I·L·D(전체, 이번 기간 이득이 양수인 상태)와 규칙별 끝까지 차이를 반복 군집 표준오차와 t(군집 수 − 1)로 다시 잰다.
결과 result/runs/check_dev_cluster_ci.json. 실행: src/에서 python -m result.check_dev_ci
"""
import json
import numpy as np
from scipy.stats import t as tdist


# 평균과 반복 군집 95% 구간 반폭 (군집 수 G, 자유도 G − 1)
def cci(v, g):
  u, n = np.unique(g), len(v)
  e = np.array([(v[g == k] - v.mean()).sum() for k in u])
  se = np.sqrt(len(u) / (len(u) - 1) * (e ** 2).sum()) / n
  return dict(mean=float(v.mean()), ci=float(tdist.ppf(0.975, len(u) - 1) * se), ci_iid=float(1.96 * v.std(ddof=1) / np.sqrt(n)), n=int(n), G=int(len(u)))


# 파일 하나: m별 (전체·이득 양수 상태) I·L·D, 규칙별 끝까지 차이
def one(name):
  rows = json.load(open(f"result/runs/{name}"))["rows"]
  g = np.array([r["rep"] for r in rows])
  I, R = np.array([r["I"] for r in rows]), np.array([r["R"] for r in rows])
  Gs, Gm = np.array([r["G_saa"] for r in rows]), np.array([r["G_mean"] for r in rows])
  I, D = I - I[:, :1], (R - R[:, :, :1]).mean(1)
  L, n = D - I, len(rows)
  out = dict(all={}, pos={}, rules={})
  for i in range(1, I.shape[1]):
    k = I[:, i] > 0
    out["all"][i] = {a: cci(x[:, i], g) for a, x in (("I", I), ("L", L), ("D", D))}
    out["pos"][i] = dict(share=float(k.mean()), **{a: cci(x[k, i], g[k]) for a, x in (("I", I), ("L", L), ("D", D))})
  pick = lambda j: D[np.arange(n), j]
  i_saa, dh = Gs.argmax(1), Gs.max(1) - Gs[:, 0]
  sel = dict(mean=Gm.argmax(1), saa_tau0=i_saa, saa_tau30=np.where(dh >= 30, i_saa, 0), oracle_now=I.argmax(1),
             oracle_small=np.where(I[:, 1] > 0, 1, 0))
  out["rules"] = {a: dict(dev=float((j > 0).mean()), **cci(pick(j), g)) for a, j in sel.items()}
  return out


def main():
  out = {s: one(f) for s, f in (("x1", "check_dev.json"), ("x2", "check_dev_x2.json"))}
  json.dump(out, open("result/runs/check_dev_cluster_ci.json", "w"), indent=1)
  for s, o in out.items():
    print(s, "pos D:", [(round(o["pos"][i]["D"]["mean"]), round(o["pos"][i]["D"]["ci"]), round(o["pos"][i]["D"]["ci_iid"])) for i in (1, 2, 3)])
    print(s, "rules:", {a: (round(v["mean"], 1), round(v["ci"], 1)) for a, v in o["rules"].items()})


if __name__ == "__main__":
  main()
