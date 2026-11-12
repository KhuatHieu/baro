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

Chọn version:

```python
from baro.root_cause_analysis import robust_scorer
robust_scorer(df, inject_time=t, version="2")       # DataFrame (FSE CSV)
robust_scorer(metrics, inject_time=t, version="2")  # dict (RCAEval JSON)
```

`run_bench.py --methods baro,baro_v2` chạy cả hai trên dữ liệu RCAEval.

### Bên RCAEval (fork `KhuatHieu/RCAEval`, commit `75c2a21`)

BARO được dùng làm tool `rank_anomalies` của agent LLM. Mỗi prediction ghi cả hai version.

| Thành phần | Version | Nội dung | Tương ứng ở đây |
|---|---|---|---|
| Toolkit | `1` | `rank_anomalies(metrics)` = BARO y hệt `RCAEval.e2e.baro`. | scorer `1` (ranking khớp 90/90 case RE2-OB) |
| Toolkit | `2` (mặc định) | Sửa F1, chỉ cho modality metrics. | scorer `2` (score khớp 90/90; xem "Kiểm tra chéo") |
| Prompt | `2` | Kiểm tra 3-5 nghi phạm bằng bằng chứng riêng, không tin một ranking. | - |
| Prompt | `nobaro` | Như v2 nhưng bỏ `rank_anomalies`. | - |

Chạy bench agent: `script/bench_llm.sh` trong RCAEval (`ARM=baro|nobaro TOOLKIT_VERSION=1|2 REP=n`).

## Cấu trúc thư mục findings

```
FINDINGS.md                         # file này
findings/<ngày>-<dataset>-<mã>/      # bằng chứng của từng finding
  runs/percase_*.jsonl              # kết quả từng case (rcaeval-agent evaluate --per-case-out)
  runs/summary_*.json               # tổng hợp + usage (chi phí, tool call)
scripts/compare_runs.py             # so sánh các run: bảng theo fault, McNemar, gộp rep
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

---

## Ghi chú vận hành

- **Lỗi tạm thời của provider làm dừng cả batch (2026-11-11, bên RCAEval).** Chạy 4 lượt song
  song (16 request đồng thời), 3/4 lượt chết vì `Connection reset by peer` và Cloudflare `520`
  từ `api.openai.com`. Case đã xong vẫn được ghi; chạy lại cùng lệnh sẽ bỏ qua chúng. Sửa:
  `num_retries=4` trong `RCAEval/agent/react.py`.
- **LiteLLM 1.109 gọi `/v1/responses` cho `openai/gpt-6-luna`**, dù client gọi
  `litellm.completion` kiểu chat. Client không truyền `reasoning_effort`, nên mức thinking là
  mặc định của model; nên ghim tường minh nếu sau này so sánh theo trục reasoning.
