# MTGNN — US Industrial Landscape (L=12, H=24)

미국 산업 패널(10개 업종 × 10개 지표)에 대한 다변량 다단계 예측.
최종 비교 세팅은 **lookback L=12개월, horizon H=24개월**이다.

기반 아키텍처: Wu et al., *Connecting the Dots*, KDD 2020
([arXiv:2005.11650](https://arxiv.org/abs/2005.11650)).

---

## 최종 모델

Small **HGRU-CTX+HE** (채널 32/32/64/128, `horizon_emb_dim=8`, `lr=2e-3`, MAE).

| 이름 | 설명 |
|------|------|
| MTGNN HGRU-CTX+HE | 업종 그래프 + horizon GRU decoder |
| MTGNN HGRU-CTX+HE FGraph | 위에 factorized feature graph stem (`A_I`, `A_F`) |
| B-MTGNN + FGraph (MC=30) | 같은 FGraph 체크포인트, 추론 시 MC-Dropout 평균 |

베이스라인: Naive-Last, Naive-Seasonal, ARIMA, VAR, LSTM, TCN, PatchTST, LightGBM.

평가 프로토콜은 모두 **`--refit_trainval` / `fit_on=trainval`**.
`val`에 COVID 충격이 들어가므로 train-only scaler는 test 입력을 왜곡한다.

---

## 결과 (5-seed, normalised RMSE)

| model | h=3 | h=6 | h=12 | h=24 |
|-------|-----|-----|------|------|
| **PatchTST** | **0.670** | **0.752** | 0.927 | 1.176 |
| ARIMA | 0.709 | 0.842 | 1.122 | 1.543 |
| Naive-Last | 0.720 | 0.821 | 1.052 | 1.338 |
| MTGNN (no FGraph) | 0.902 | 0.878 | **0.925** | 1.067 |
| MTGNN+FGraph | 0.912 | 0.924 | 1.003 | 1.024 |
| B-MTGNN+FGraph | 0.915 | 0.921 | 0.996 | **1.022** |
| LSTM | 0.983 | 0.978 | 0.971 | 1.080 |
| TCN | 0.911 | 0.942 | 0.983 | 1.160 |
| LightGBM | 1.887 | 1.839 | 1.763 | 1.754 |

단기(h≤6)는 PatchTST, 장기(h=24)는 FGraph / B-MTGNN이 위 목록 중 가장 낮다.
상세 숫자·MAPE·시드별 JSON은 `results/`.

---

## 설치

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
```

NVIDIA GPU가 있으면 `torch`를 CUDA 휠로 먼저 깐 뒤 나머지를 설치하면 된다.

```powershell
.\.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu124
.\.venv\Scripts\pip install -r requirements.txt
```

---

## 한 줄 재현

프로젝트 루트에서:

```powershell
.\.venv\Scripts\python.exe prepare_industrial.py
.\.venv\Scripts\python.exe run_experiment.py --device cuda:0
```

체크포인트가 이미 있으면 재학습을 건너뛴다 (`--skip_train`).
시드 JSON을 다시 집계만 하려면 `python summarize.py`.

원시 패널을 BLS에서 다시 받으려면:

```powershell
.\.venv\Scripts\python.exe build_US_Industrial_Landscape_full.py
```

출력은 `US_Industrial_Landscape_complete_tensor.npz`. 재수집이 아니면 루트의 complete 파일을 그대로 쓰면 된다.

---

## 구성

```
prepare_industrial.py     sliding-window split (default L=12, H=24)
train_industrial.py       MTGNN 학습 (--feature_graph 로 FGraph on/off)
eval_mtgnn.py             MTGNN / B-MTGNN 평가 (n_mc=0 또는 30)
eval_baselines.py         Naive / ARIMA / VAR / LSTM / TCN / PatchTST / LightGBM
run_experiment.py         5-seed 전체 비교
summarize.py              results/summary_L12_H24.json
net.py  layer.py          HGRU-CTX+HE ± feature graph
plot_predictions.py
plot_adjacency.py
plot_all_predictions.py
data/industrial_L12_H24/
results/seed{1-5}_L12_H24.json
results/summary_L12_H24.json
checkpoints/industrial/   L=12 H=24 최종 weight
```

체크포인트 이름:

- `mtgnn_s{seed}_L12_H24.pth` — no FGraph
- `mtgnn_fgraph_s{seed}_L12_H24.pth` — FGraph (B-MTGNN도 이 파일 사용)
- `lstm_` / `tcn_` / `patchtst_` / `lightgbm_` `L12_H24_trval_s{seed}`

예측·인접행렬 그림:

```powershell
.\.venv\Scripts\python.exe plot_predictions.py
.\.venv\Scripts\python.exe plot_adjacency.py
.\.venv\Scripts\python.exe plot_all_predictions.py
```

---

## Citation

```bibtex
@inproceedings{wu2020connecting,
  title={Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks},
  author={Wu, Zonghan and Pan, Shirui and Long, Guodong and Jiang, Jing and Chang, Xiaojun and Zhang, Chengqi},
  booktitle={Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  year={2020}
}
```
