# FINDINGS: BARO

Nhật ký các phát hiện về BARO (RobustScorer) khi dùng trên RCAEval, kể cả khi BARO là tool
của một agent LLM. Mỗi mục ghi: hiện tượng, nguyên nhân, cách sửa, version thay đổi, kết quả
trước/sau, và bằng chứng để tra lại.

## Bảng version

### Repo này (`baro`)

| Thành phần | Version | Nội dung | Ở đâu |
|---|---|---|---|
| RobustScorer | `1` (mặc định) | BARO như paper. **Có lỗi F1.** Giữ làm mặc định để kết quả paper tái lập được. | `baro/root_cause_analysis.py` |
| RobustScorer | `2` | Sửa F1: giữ series phẳng trước inject, không chấm theo đơn vị thô. | `baro/root_cause_analysis.py` (`robust_score`) |
| RobustScorer | `3` (thử nghiệm) | F2: điểm như v2, xếp theo 3 bậc: bước nhảy kéo dài từ nền lặng → z-score → series lặng không có bước nhảy. | `baro/root_cause_analysis.py` (`sustained_step`, `is_quiet`, `rank_scored`) |

Chọn version:

```python
from baro.root_cause_analysis import robust_scorer
robust_scorer(df, inject_time=t, version="2")       # DataFrame (FSE CSV)
robust_scorer(metrics, inject_time=t, version="2")  # dict (RCAEval JSON)
```

`run_bench.py --methods baro,baro_v2,baro_v3` chạy các phiên bản trên dữ liệu RCAEval.
Đánh giá offline (không LLM): `scripts/eval_ranking.py` (metrics) và `scripts/eval_multisource.py`
(logs/traces); cả hai cần RCAEval đã cài và một task view (`rcaeval-agent export`).

### Bên RCAEval (fork `KhuatHieu/RCAEval`, commit `75c2a21`)

BARO được dùng làm tool `rank_anomalies` của agent LLM. Mỗi prediction ghi cả hai version.

| Thành phần | Version | Nội dung | Tương ứng ở đây |
|---|---|---|---|
| Toolkit | `1` | `rank_anomalies(metrics)` = BARO y hệt `RCAEval.e2e.baro`. | scorer `1` (ranking khớp 90/90 case RE2-OB) |
| Toolkit | `2` (mặc định) | Sửa F1, chỉ cho modality metrics. | scorer `2` (score khớp 90/90; xem "Kiểm tra chéo") |
| Toolkit | `3` (thử nghiệm) | F2, chỉ cho metrics; series có bước nhảy kèm nhãn `step`. Logs/traces như v1. | scorer `3` (ranking khớp 215/215 case RE2-OB + RE1-OB) |
| Prompt | `2` | Kiểm tra 3-5 nghi phạm bằng bằng chứng riêng, không tin một ranking. | - |
| Prompt | `nobaro` | Như v2 nhưng bỏ `rank_anomalies`. | - |

Chạy bench agent: `script/bench_llm.sh` trong RCAEval (`ARM=baro|nobaro TOOLKIT_VERSION=1|2|3 REP=n`).

## Cấu trúc thư mục findings

```
FINDINGS.md                         # file này
findings/<ngày>-<dataset>-<mã>/      # bằng chứng của từng finding
  runs/percase_*.jsonl              # kết quả từng case (rcaeval-agent evaluate --per-case-out)
  runs/summary_*.json               # tổng hợp + usage (chi phí, tool call)
  offline/<dataset>.txt, .jsonl     # đánh giá offline của các scorer (eval_ranking.py)
scripts/compare_runs.py             # so sánh các run LLM: bảng theo fault, McNemar, gộp rep
scripts/eval_ranking.py             # đánh giá offline thứ hạng metric đúng theo scorer
scripts/eval_multisource.py         # như trên cho logs / trace_latency / trace_errors
scripts/eval_v3_nolatstep.py        # thử post-hoc: v3 nhưng bậc 1 bỏ series *_latency
```

Trajectory và prediction đầy đủ (nặng) nằm ở `RCAEval/results/` (không commit).

---

## F1. BARO làm agent LLM kém đi: che mất tín hiệu disk/socket (2026-11-11)

**Setup:** RCAEval RE2-OB, 90 case (6 loại fault × 15), agent tham chiếu của RCAEval,
`openai/gpt-6-luna` (tier rẻ), temperature 0, max 30 steps. Bằng chứng:
`findings/2026-11-11-re2ob-f1/`.

### Hiện tượng

Agent LLM **không có** tool BARO tốt hơn agent **có** BARO:

| Run | service AC@1 | indicator AC@1 | indicator Avg@5 | $/case | tool calls/case |
|---|---|---|---|---|---|
| LLM + BARO (scorer v1) | 0.97 | 0.56 | 0.74 | 0.0029 | 30.4 |
| LLM không BARO | 1.00 | 0.70 | 0.86 | 0.0042 | 41.9 |
| BARO đứng riêng (v1) | 0.14 | 0.09 | 0.39 | - | - |

Paired: indicator AC@1 lệch 6 vs 19 case (McNemar p = 0.015). Service AC@1 đã chạm trần
(lệch 0 vs 3, p = 0.25), nên trên RE2-OB metric này không phân biệt được các arm.

Toàn bộ chênh lệch dồn vào hai loại fault:

| fault | indicator AC@1, LLM + BARO v1 | LLM không BARO | metric đúng `<root>_<indicator>` có trong output BARO v1 |
|---|---|---|---|
| disk | 5/15 | 14/15 | **0/15** |
| socket | 0/15 | 2/15 | **0/15** |
| 4 loại còn lại (60 case) | 45/60 | 47/60 | 57/60 |

Bỏ disk và socket ra thì hai arm ngang nhau (lệch 6 vs 8 case, p ≈ 0.8).

### Nguyên nhân

1. **BARO bỏ series phẳng trước inject.** Đường DataFrame của `robust_scorer` gọi
   `preprocess()` (có `drop_constant`) riêng cho nửa normal và nửa anomalous, rồi chỉ giữ cột
   có ở cả hai nửa. Metric hằng số trước inject bị xoá dù sau inject nó nhảy mạnh, mà đó lại là
   tín hiệu rõ nhất. Ví dụ:
   - `re2-ob-0024 productcatalogservice_diskio`: trước = 0, sau ≈ 9.4e9, bị loại.
   - `re2-ob-0001 currencyservice_socket`: trước = 4, sau = 17-19, bị loại.

   Tình huống này xảy ra ở **mọi** case disk và socket của RE2-OB. Đây cũng là lý do BARO đứng
   riêng có indicator AC@1 = 0/15 cho cả disk lẫn socket.
2. **IQR = 0 làm score tính theo đơn vị thô.** Với series bursty mà phần lớn bằng 0 (IQR = 0),
   `RobustScaler` dùng scale = 1, nên score tính bằng byte. `redis_diskio` (nhiễu, lần nào cũng
   nhảy) vì thế đứng **top-1 trong 78/90 case**. Đường dict (`robust_scorer_dict`) có cùng lỗi
   này qua `if iqr == 0: iqr = 1`, dù nó không bỏ series phẳng.
3. **LLM tier rẻ bị anchoring theo ranking.** Với BARO, agent chỉ mở `get_metric(<root>_diskio)`
   ở 5/15 case disk (không BARO: 14/15). Nó còn tự gạt diskio đi vì thấy `redis_diskio` nhiễu
   ("Redis disk I/O is bursty… does not track the incident", re2-ob-0024).

Kết luận đúng là: **BARO v1 che mất tín hiệu disk/socket và đẩy nhiễu lên đầu**. Không phải
"LLM không cần anomaly ranking".

### Cách sửa: scorer v2

`robust_score(normal, anomal, window)` trong `baro/root_cause_analysis.py`:

- Bỏ series hằng số trên **toàn cửa sổ** (không theo từng nửa); chuyển mem sang MB như cũ.
- Score = `max(anomal - median(normal)) / scale`, trong đó `scale` lần lượt là: IQR(normal);
  nếu ≈ 0 thì std(normal); nếu vẫn ≈ 0 (phẳng trước inject) thì 1% range của series trên
  cửa sổ. Một bước nhảy từ nền phẳng vì thế có score ≈ 100 bất kể đơn vị.
- "≈ 0" nghĩa là ≤ 1e-9 × max|series|: IQR của một hằng số có thể ra ~1e-18 do sai số float.
- Series có IQR > 0 được chấm **giống hệt** BARO v1. NaN được bỏ qua.
- Đường dict v2 dùng cùng `robust_score`, nên chỉ tính chiều tăng như đường DataFrame. v1 của
  đường dict dùng `abs()`, tức tính cả chiều giảm; đây là khác biệt có chủ ý.

Kiểm tra offline (không LLM) trên 90 case: thứ hạng của `<root>_<indicator>` trong ranking.

| | ind@1 | ind@5 | ind@20 (top_k mặc định của tool) | service top-1 | service top-3 |
|---|---|---|---|---|---|
| scorer v1 | 8 | 52 | 57 | 13 | 79 |
| scorer v2 | 33 | 85 | **90** | 77 | 87 |

Độ nhạy của ngưỡng nền phẳng (được chọn trên chính 90 case này, cần lưu ý; sweep chạy trước
khi thêm tolerance float nên số tuyệt đối hơi khác bảng trên): với fallback std, 1% cho
ind@5 = 84 và 5% cho ind@5 = 63 (disk tụt xuống top-5 chỉ còn 3/15, vì socket nhiễu của các
service khác cũng ≈ 1/fraction). Ở **mọi** giá trị đã thử (0.5%, 1%, 5%), ind@20 = 90/90.

Test: `tests/test.py` (`test_robust_score_does_not_score_in_raw_units`,
`test_scorer_version_1_stays_the_default`, `test_scorer_version_2_keeps_series_flat_before_the_inject_time`,
`test_scorer_dict_version_2`).

### Kiểm tra chéo với RCAEval

Trên 90 case RE2-OB, cùng dữ liệu đầu vào (`CaseToolkit.standard_metrics()`):

- scorer v1 ở đây và toolkit v1 của RCAEval: **ranking giống hệt 90/90**.
- scorer v2 ở đây và toolkit v2 của RCAEval: **score giống hệt 90/90** (so ở 3 chữ số thập
  phân). Ranking giống hệt 71/90; 19 case còn lại chỉ khác thứ tự **giữa các series hoà điểm**
  (thường là nhiều series ≈ 100): toolkit làm tròn score rồi mới sắp xếp, còn ở đây sắp xếp
  trên giá trị chưa làm tròn.

### Kết quả với agent LLM sau khi sửa (2026-11-11)

Mỗi cấu hình chạy 2 lần (gpt-6-luna, temperature 0). "tk" = toolkit version bên RCAEval.

| Run | service AC@1 | ind AC@1 | ind AC@3 | ind Avg@5 | cpu | mem | disk | socket | delay | loss | $/case | calls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LLM + BARO tk1 r1 | 87 | 50 | 71 | 0.742 | 12 | 12 | 5 | 0 | 14 | 7 | 0.0029 | 30.4 |
| LLM + BARO tk1 r2 | 89 | 53 | 71 | 0.738 | 12 | 13 | 3 | 0 | 15 | 10 | 0.0028 | 30.2 |
| LLM + BARO tk2 r1 | 89 | 57 | 81 | 0.847 | 11 | 13 | 6 | 1 | 13 | 13 | 0.0026 | 28.9 |
| LLM + BARO tk2 r2 | 89 | 57 | 78 | 0.836 | 12 | 11 | 7 | 1 | 14 | 12 | 0.0026 | 28.7 |
| LLM không BARO r1 | 90 | 63 | 81 | 0.858 | 14 | 8 | 14 | 2 | 15 | 10 | 0.0042 | 41.9 |
| LLM không BARO r2 | 90 | 60 | 81 | 0.847 | 12 | 11 | 11 | 1 | 14 | 11 | 0.0041 | 40.8 |

(Cột cpu…loss là indicator AC@1, mỗi loại /15.)

**Nhiễu giữa các lần chạy là lớn.** Dù temperature = 0, hai lần chạy cùng cấu hình vẫn lệch
nhau ở 11-15/90 case về indicator AC@1. Vì vậy dưới đây là so sánh gộp các rep (điểm mỗi case
= trung bình 2 rep; sign-flip permutation test paired theo case; bootstrap 95% CI):

| So sánh | ind AC@1 (B − A) | p | ind Avg@5 (B − A) | p |
|---|---|---|---|---|
| tk1 → tk2 (tác dụng của bản sửa) | +0.061 [−0.011, +0.139] | 0.16 | **+0.101 [+0.041, +0.167]** | **0.002** |
| tk2 → không BARO | +0.050 [−0.044, +0.144] | 0.36 | +0.011 [−0.043, +0.068] | 0.73 |
| tk1 → không BARO (F1 ban đầu, có lặp) | **+0.111 [+0.028, +0.200]** | **0.018** | **+0.112 [+0.048, +0.181]** | **0.001** |

**Kết luận:**

1. F1 được xác nhận sau khi có lặp: với BARO v1, agent không có BARO tốt hơn rõ ràng.
2. Scorer v2 cải thiện có ý nghĩa thống kê ở indicator Avg@5 (+0.10, p = 0.002). AC@1 tăng
   (+0.06) nhưng chưa đủ ý nghĩa với 90 case × 2 rep.
3. Sau khi sửa, **LLM + BARO và LLM không BARO không còn khác nhau có ý nghĩa** (Avg@5 chênh
   0.01, p = 0.73), trong khi arm có BARO **rẻ hơn ~37%** ($0.0026 so với $0.0041/case) và ít
   tool call hơn (29 so với 41).
4. Phần chênh còn lại tập trung ở fault **disk** (gộp: 6.5 so với 12.5 /15). Agent giờ đã thấy
   và mở `<root>_diskio` (13/15 case, so với 5/15 với v1), nhưng vẫn xếp `mem` lên trước. Trong
   RE2-OB, fault disk cũng đẩy memory lên tới trần (vd. 44 → 268 MB, có lẽ do page cache).
   `mem` có z-score cỡ hàng nghìn, còn `diskio` tăng từ đúng 0 nên theo thiết kế chỉ ≈ 100.
   Model bám theo thứ tự này.
5. Độ lớn mẫu: CI của hiệu ứng AC@1 rộng khoảng ±0.09; các khác biệt nhỏ hơn mức đó không
   phát hiện được với 90 case × 2 rep. Cần thêm rep (~$0.25/run) hoặc thêm dataset (RE2-TT).

Tái lập bảng trên từ bằng chứng:

```
R=findings/2026-11-11-re2ob-f1/runs
python scripts/compare_runs.py \
  tk1_r1=$R/percase_llm_baro_tk1_r1.jsonl tk1_r2=$R/percase_llm_baro_tk1_r2.jsonl \
  tk2_r1=$R/percase_llm_baro_tk2_r1.jsonl tk2_r2=$R/percase_llm_baro_tk2_r2.jsonl \
  nobaro_r1=$R/percase_llm_nobaro_r1.jsonl nobaro_r2=$R/percase_llm_nobaro_r2.jsonl \
  baro_alone=$R/percase_baro_alone.jsonl \
  --groups tk1=tk1_r1,tk1_r2 tk2=tk2_r1,tk2_r2 nobaro=nobaro_r1,nobaro_r2 \
  --group-pairs tk1:tk2 tk2:nobaro tk1:nobaro
```

### Còn mở (ứng viên v3)

- **Fault disk:** cho series tăng từ nền phẳng một score cao hơn, để `<root>_diskio` không
  đứng sau `mem` (kết luận 4). Phải làm sao không đẩy nhiễu socket (3 → 4) lên cùng: thử xét
  độ bền của bước nhảy (giữ suốt giai đoạn sau inject hay chỉ thoáng qua).
- Nhiều series phẳng trước inject cùng ra ≈ 100 và hoà điểm với root cause. Cần tie-break,
  ví dụ theo độ thay đổi tương đối; đồng thời thống nhất cách tie-break với toolkit RCAEval.
- **Logs/traces (multi-source BARO) cũng có vấn đề IQR = 0 → đơn vị thô**, nhưng áp dụng
  `robust_score` cho trace_latency làm kết quả **tệ hơn** offline (service top-1: 29 → 19):
  series 15 s bị lượng tử hoá, nhiều series phẳng trước inject cùng ≈ 100 và mất thông tin độ
  lớn. Logs thì tốt hơn (2 → 8). Chưa đưa vào v2.
- **Fault socket:** mọi arm chỉ 0-2/15. `socket` đã có trong ranking v2 nhưng agent vẫn chọn
  `latency`; cần đọc trajectory.
- Tham số v2 được chọn trên RE2-OB. Quy ước cho v3: thiết kế trên RE2-OB, kiểm chứng trên RE2-TT
  trước khi đổi mặc định. Service-level đã chạm trần trên RE2-OB.

## F2. Scorer v3: xếp bậc theo loại tín hiệu (2026-11-12; tốt trên RE2-OB, thất bại trên RE2-TT)

**Mục tiêu:** sửa hai điểm còn lại sau F1: (a) fault disk, `<root>_diskio` đứng sau `mem`;
(b) fault socket gần như luôn sai. Bằng chứng: `findings/2026-11-12-f2/`.

### Quan sát dẫn tới thiết kế

Trên RE2-OB, xét các series phẳng trước inject (v2 cho chúng cùng score ≈ 100):

| Điều kiện | Metric root cause (15 diskio + 15 socket) | Series nhiễu |
|---|---|---|
| Khác nền ở ≥ 90% mẫu sau inject | 30/30 | 36/145 |
| ... và median tăng ≥ 50% so với nền | 30/30 | 21/145 (18 trong số đó trên chính service gốc) |

"Bước nhảy kéo dài từ nền phẳng" là tín hiệu rất đặc hiệu.

### Phát hiện phụ: v2 kém hơn v1 trên RE1-OB

Kiểm chứng trên RE1-OB (125 case, chỉ có metrics) cho thấy **v2 kém v1**: ind@1 78 → 71,
svc@1 92 → 82. Nguyên nhân là fallback std của v2 khuếch đại các series gần như hằng số:

| Case | Series | std trước inject | thay đổi sau | v2 score | v1 (IQR 0 → scale 1) |
|---|---|---|---|---|---|
| re1-ob-0006 | `emailservice_mem` | 0.0006 MB | +0.2 MB | 316 | 0.2 |
| re1-ob-0096 | `paymentservice_latency` | 1.6e-5 s | +0.012 s | 733 | 0.012 |

v1 "đúng một cách tình cờ": scale = 1 trong đơn vị MB/giây gần như đẩy các series lặng
(IQR = 0) xuống cuối; nó chỉ sai khi đơn vị lớn (byte của diskio). Kết luận: **z-score chỉ có
nghĩa khi giai đoạn normal có độ phân tán.**

### Thiết kế v3

Điểm số như v2, xếp theo 3 bậc (`rank_scored`):

1. **Bước nhảy kéo dài** (`sustained_step`): normal lặng (IQR ≈ 0, nền = median, cho phép vài
   dao động lẻ), khác nền ở ≥ 90% mẫu sau inject, median sau ≥ 2 lần nền (`min_rise=1.0`;
   tăng từ 0 tính là vô hạn). Trong bậc: rise lớn trước.
2. Series có độ phân tán trước inject, theo z-score (như BARO).
3. Series lặng không có bước nhảy, theo score fallback.

Toolkit v3 (RCAEval) gắn thêm nhãn cho bậc 1, vd. `"step": "from 0 to 8.234e+09, held in 96%
of samples after the inject time"`.

### Đánh giá offline (không LLM)

`offline/re2-ob.txt`, `offline/re1-ob.txt`. ind@k = số case metric đúng có hạng ≤ k.

| Scorer | RE2-OB ind@1 | ind@3 | ind@20 | svc@1 | svc@3 | RE1-OB ind@1 | ind@3 | svc@1 |
|---|---|---|---|---|---|---|---|---|
| v1 | 8 | 39 | 57 | 13 | 79 | **78** | **103** | **92** |
| v2 | 33 | 64 | 90 | **77** | **87** | 71 | 96 | 82 |
| **v3** | **55** | **82** | **90** | 76 | 85 | **78** | **103** | **92** |

Theo fault (RE2-OB ind@1, v2 → v3): disk 0 → **15/15**, socket 1 → **15/15**, delay 11 → 11,
cpu 2 → 2, nhưng loss 10 → 6 và mem 9 → 6: `socket` của chính service gốc (triệu chứng đi kèm)
cũng có bước nhảy và vượt lên metric đúng.

Ablation (RE2-OB ind@1 / RE1-OB ind@1):

| Biến thể | RE2-OB | RE1-OB | Ghi chú |
|---|---|---|---|
| v3 (mặc định) | 55 | 78 | |
| bỏ bậc 3 (`quiet_rank=score`) | 55 | 71 | bậc 3 là thứ khôi phục RE1-OB |
| bỏ bậc 1 (`step_rank=none`) | 29 | 78 | bậc 1 là thứ sửa disk/socket |
| nền phẳng tuyệt đối (`baseline=flat`) | 54 (ind@20: 86) | 78 | 4 latency đúng bị đẩy xuống bậc 3 |
| `min_rise=0.5` | 53 | 78 | thêm 14 cờ nhầm trên service gốc |

**Cảnh báo overfit:** ngưỡng và cách xếp bậc được chọn khi nhìn RE2-OB, và RE1-OB đã được dùng
để chẩn đoán bậc 3, nên cả hai không còn là tập kiểm chứng độc lập. RE2-TT là tập kiểm chứng
thật (chưa nhìn).

**Logs/traces (kết quả âm tính):** áp cùng cách xếp bậc cho logs/trace_latency/trace_errors
làm kém đi (trace_latency svc@3: v1 55, v2 48, v3 32; logs svc@3: v1 16, v2 24, v3 11;
`offline/re2-ob_multisource.txt`). Toolkit v3 giữ logs/traces như v1.

### Kiểm chứng trên RE2-TT (2026-11-12): **v3 không tổng quát hoá**

RE2-TT (Train Ticket, 90 case, 6 fault × 15) là tập chưa từng nhìn khi thiết kế v3.
`offline/re2-tt.txt`, `offline/re2-tt.jsonl`.

| Scorer | ind@1 | ind@3 | ind@5 | ind@20 | svc@1 | svc@3 | cpu | delay | disk | loss | mem | socket |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v1 | **36** | **50** | **57** | **76** | **60** | **74** | 1 | 7 | 15 | 8 | 5 | 0 |
| v2 | **36** | 47 | 53 | 74 | **60** | 71 | 1 | 7 | 15 | 8 | 5 | 0 |
| v3 | 18 | 48 | 57 | 75 | 29 | 71 | 1 | 0 | 12 | 3 | 2 | 0 |
| v3 `step_rank=none` | 36 | 50 | 57 | 75 | 60 | 74 | 1 | 7 | 15 | 8 | 5 | 0 |

(cpu…socket: ind@1, mỗi loại /15.)

**v3 làm svc@1 giảm một nửa (60 → 29).** Bậc 1 gắn cờ 45 series, **0 series là target**:

- `ts-admin-travel-service_latency` bị gắn cờ ở 34/90 case. Trước inject nó đứng yên ở đúng
  0.235 (IQR ≈ 0, 2 giá trị duy nhất; có lẽ endpoint admin gần như không có request nên giá trị
  cũ được giữ lại). Sau inject nó dao động (0.4-9.5) ở ≥ 93% mẫu, **ở mọi loại fault**. Đây là
  artifact của workload chứ không phải tín hiệu fault, nhưng nó thoả đúng định nghĩa "bước nhảy
  kéo dài từ nền lặng".
- Phần còn lại: `ts-preserve-service_latency` (7), `ts-preserve-other-service_latency` (2),
  `ts-inside-payment-service_latency` (2), cùng kiểu.

Thêm vài điểm về RE2-TT:
- Fault disk đã đúng 15/15 với v1: `diskio` không phẳng trước inject như ở RE2-OB, nên lỗi F1
  không xuất hiện ở đây. v2 = v1 ở ind@1.
- Socket bằng 0/15 với mọi version, thứ hạng của target khoảng 5-16 và không bị gắn cờ step:
  đây là vấn đề khác, không liên quan v3.

**Thử post-hoc (đã nhìn RE2-TT, không còn là kiểm chứng độc lập):** loại series `*_latency`
khỏi bậc 1 (`scripts/eval_v3_nolatstep.py`, patch `rank_scored`):

| | RE2-TT ind@1 / svc@1 | RE2-OB ind@1 / ind@20 / svc@1 | RE1-OB ind@1 / svc@1 |
|---|---|---|---|
| v3 | 18 / 29 | 55 / 90 / 76 | 78 / 92 |
| v3 bỏ latency khỏi bậc 1 | 36 / 60 | 53 / 86 / 75 | 78 / 92 |

Biến thể này đưa RE2-TT về ngang v1 và giữ gần hết lợi ích trên RE2-OB (mất 4 case latency
khỏi top-20, giống ablation `baseline=flat`). Tuy vậy nó được chọn sau khi nhìn cả 3 dataset,
nên **cần một tập kiểm chứng mới** (RE2-SS hoặc RE3-*) trước khi đưa vào code.

**Hệ quả:**
1. Không đổi mặc định sang v3. Toolkit v3 giữ nhãn "thử nghiệm".
2. Không bench LLM tk3 trên RE2-TT: bước 2 của "Việc tiếp theo" có điều kiện "nếu RE2-TT giữ
   được kết quả", và điều kiện này không thoả. Offline đã cho thấy tk3 sẽ đẩy nhiễu lên đầu.
3. Kết quả bench RE2-OB bên dưới (tk3 > không BARO) chỉ đúng trên dataset dùng để thiết kế v3.

### Bench LLM (RE2-OB, 3 rep mỗi cấu hình)

Hoàn tất 2026-11-12 (đã chạy tiếp từ phần dở; cả 9 run đủ 90/90 case, không trùng case).
gpt-6-luna, temperature 0, max 30 steps. Bằng chứng: `findings/2026-11-12-f2/runs/` (tk2 r1-r2 và
nobaro r1-r2 dùng lại file của F1), bảng đầy đủ: `findings/2026-11-12-f2/compare_3rep.txt`.

| Run | svc AC@1 | ind AC@1 | ind AC@3 | ind Avg@5 | cpu | mem | disk | socket | delay | loss | $/case | calls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tk2 r1 | 89 | 57 | 81 | 0.847 | 11 | 13 | 6 | 1 | 13 | 13 | 0.0026 | 28.9 |
| tk2 r2 | 89 | 57 | 78 | 0.836 | 12 | 11 | 7 | 1 | 14 | 12 | 0.0026 | 28.7 |
| tk2 r3 | 89 | 56 | 82 | 0.847 | 12 | 11 | 5 | 2 | 14 | 12 | 0.0026 | 28.5 |
| **tk3 r1** | 89 | 68 | 88 | 0.927 | 10 | 9 | 15 | 8 | 15 | 11 | 0.0024 | 28.9 |
| **tk3 r2** | 89 | 64 | 90 | 0.922 | 12 | 7 | 14 | 6 | 14 | 11 | 0.0026 | 29.3 |
| **tk3 r3** | 90 | 75 | 88 | 0.944 | 12 | 12 | 15 | 8 | 15 | 13 | 0.0025 | 29.5 |
| không BARO r1 | 90 | 63 | 81 | 0.858 | 14 | 8 | 14 | 2 | 15 | 10 | 0.0042 | 41.9 |
| không BARO r2 | 90 | 60 | 81 | 0.847 | 12 | 11 | 11 | 1 | 14 | 11 | 0.0041 | 40.8 |
| không BARO r3 | 90 | 61 | 82 | 0.862 | 13 | 10 | 13 | 1 | 15 | 9 | 0.0041 | 42.4 |

Gộp 3 rep (điểm mỗi case = trung bình 3 run; cpu…loss là số case đúng trung bình /15):

| nhóm | ind AC@1 | ind Avg@5 | cpu | mem | disk | socket | delay | loss |
|---|---|---|---|---|---|---|---|---|
| tk2 | 0.630 | 0.843 | 11.7 | 11.7 | 6.0 | 1.3 | 13.7 | 12.3 |
| **tk3** | **0.767** | **0.931** | 11.3 | 9.3 | **14.7** | **7.3** | 14.7 | 11.7 |
| không BARO | 0.681 | 0.856 | 13.0 | 9.7 | 12.7 | 1.3 | 14.7 | 10.0 |

| So sánh (A → B) | ind AC@1 (B − A) | p | ind Avg@5 (B − A) | p |
|---|---|---|---|---|
| tk2 → tk3 | **+0.137 [+0.052, +0.222]** | **0.004** | **+0.088 [+0.045, +0.133]** | **0.0003** |
| không BARO → tk3 (dấu đảo từ bảng gốc `tk3:nobaro`) | +0.085 [+0.004, +0.167] | 0.054 | **+0.076 [+0.036, +0.119]** | **0.0009** |
| tk2 → không BARO | +0.052 [−0.033, +0.137] | 0.28 | +0.013 [−0.036, +0.062] | 0.65 |

**Kết luận:**

1. tk3 cải thiện có ý nghĩa so với tk2 ở cả AC@1 (+0.14) và Avg@5 (+0.09).
2. **Lần đầu tiên agent có BARO tốt hơn agent không có BARO**: Avg@5 +0.076 (p < 0.001), AC@1
   +0.085 (p = 0.054, sát ngưỡng), trong khi vẫn rẻ hơn ~40% ($0.0025 so với $0.0041/case) và ít
   tool call hơn (29 so với 42).
3. Khớp dự đoán offline: disk 6.0 → 14.7, socket 1.3 → 7.3. Socket chỉ được 7.3/15 trong khi
   offline tk3 đạt 15/15 ở ind@1: agent vẫn hay chọn `latency` dù ranking đã đúng (cần đọc
   trajectory). Cái giá ở mem (11.7 → 9.3) như dự đoán.
4. Nhiễu giữa rep vẫn lớn (tk3 ind AC@1: 64-75/90). Tham số v3 được chọn trên RE2-OB, nên kết
   luận 2 cần kiểm chứng trên RE2-TT. **Offline trên RE2-TT, v3 thất bại** (xem mục trên), nên
   kết luận 2 chưa tổng quát hoá được.

### Việc tiếp theo

1. ~~Hoàn tất bench RE2-OB~~ (xong). ~~Kiểm chứng RE2-TT offline~~ (xong: v3 thất bại).
2. Tìm định nghĩa bậc 1 không bắt artifact workload. Ứng viên: chỉ metric tài nguyên (bỏ
   latency), xem bảng post-hoc ở trên. Kiểm chứng trên một dataset chưa nhìn (RE2-SS, RE3-*)
   trước khi đưa vào code; chỉ bench LLM trên RE2-TT nếu biến thể mới qua được bước này.
3. Socket: RE2-OB agent chỉ đúng 7.3/15 dù ranking đúng; RE2-TT ranking sai ở mọi version.
   Cần đọc trajectory / xem dữ liệu socket của Train Ticket.
4. Cái giá ở cpu/mem trên RE2-OB: `socket` của service gốc vượt lên metric đúng.

---

## Ghi chú vận hành

- **Lỗi tạm thời của provider làm dừng cả batch (2026-11-11, bên RCAEval).** Chạy 4 lượt song
  song (16 request đồng thời), 3/4 lượt chết vì `Connection reset by peer` và Cloudflare `520`
  từ `api.openai.com`. Case đã xong vẫn được ghi; chạy lại cùng lệnh sẽ bỏ qua chúng. Sửa:
  `num_retries=4` trong `RCAEval/agent/react.py`.
- **LiteLLM 1.109 gọi `/v1/responses` cho `openai/gpt-6-luna`**, dù client gọi
  `litellm.completion` kiểu chat. Client không truyền `reasoning_effort`, nên mức thinking là
  mặc định của model; nên ghim tường minh nếu sau này so sánh theo trục reasoning.
