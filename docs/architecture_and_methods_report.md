# Phân tích kiến trúc & phương pháp — Dự án VSL-400

> Báo cáo kỹ thuật (góc nhìn nghiên cứu viên Computer Vision), xây dựng dựa trên chính mã nguồn của dự án Vietnamese Sign Language Recognition (VSL-400).

---

## 1. Bối cảnh bài toán

Dự án nhận dạng **400 lớp** (gloss) Ngôn ngữ Ký hiệu Việt Nam từ video. Có hai *nhánh phương pháp* (modality) song song:

| Nhánh | Đầu vào | Mô hình tiêu biểu trong repo |
|------|---------|------------------------------|
| **Pose-based** (dựa trên khung xương) | Toạ độ khớp (keypoints) | **SPOTER**, SL-GCN, DSTA-SLR |
| **RGB-based** (dựa trên điểm ảnh) | Khung hình video thô | **VideoMAE**, Swin3D, R3D/R(2+1)D, S3D, MViT |

Hai mô hình đang thực sự được huấn luyện và cần so sánh là **SPOTER (pose)** và **VideoMAE (RGB)** — đại diện cho hai triết lý đối lập. Báo cáo giải thích kỹ hai mô hình này và điểm qua phần còn lại.

**Một điểm cực kỳ quan trọng về dữ liệu** (đọc từ `src/features/hf_builders/visl_400.py`): tập train/val/test được chia **theo người ký (signer-independent)**, không phải chia ngẫu nhiên. Test = người ký `024` (hoàn toàn không xuất hiện khi train). Vì vậy thước đo thật sự của dự án là **khả năng tổng quát hoá sang người ký mới**. Đây là lý do mô hình hiện tại (SPOTER đa camera, run `spoter_v3.0_multicam`) đạt val **93.7%** nhưng test chỉ **85.7%** — khoảng cách này **không phải overfitting thông thường**, mà là độ khó của generalization. Điều này quyết định "phương pháp nào tốt hơn".

---

## 2. Nhánh Pose-based: SPOTER

### 2.1. Toàn bộ pipeline đặc trưng (feature pipeline)

Đây là phần thường bị bỏ qua nhưng quyết định 50% chất lượng:

1. **Trích xuất khung xương**: `video_to_pose --format mediapipe` (MediaPipe Holistic) → file `.pose`. Mỗi frame cho ~543 điểm (thân + 2 bàn tay + mặt).
2. **Chọn khớp** (`SPOTERJointSelect`): chỉ giữ **54 điểm** = 12 điểm thân (mũi, cổ, mắt, tai, vai, khuỷu, cổ tay) + 21 điểm × 2 bàn tay. Chỉ lấy **toạ độ (x, y)** → bỏ chiều sâu z và độ tin cậy.
3. **Chuẩn hoá** (`SPOTERSingleBodyDictNormalize`, `SPOTERSingleHandDictNormalize`):
   - Thân người: chuẩn hoá theo **hộp giới hạn dựa trên độ rộng vai** (shoulder distance làm "head metric") → loại bỏ ảnh hưởng khoảng cách camera, chiều cao người.
   - Bàn tay: chuẩn hoá riêng theo bounding box của từng bàn tay → tay luôn nằm trong cùng một thang đo bất kể ở đâu trong khung hình.
   - `SPOTERShift`: dịch toạ độ về quanh 0 (trừ 0.5).
4. **Padding thời gian** (`SPOTERPad`): chuẩn hoá về **70 frame** (lặp lại chuỗi nếu ngắn).
5. **Augmentation** (chỉ khi train): xoay toàn thân ±13°, shear/squeeze ≤15%, xoay khớp tay ±4°, tuỳ chọn nhiễu Gaussian. Đây là các phép biến đổi **giữ nguyên ngữ nghĩa ký hiệu** nhưng mô phỏng biến thể khi thực hiện.

> **Nhận xét:** bước chuẩn hoá theo vai/bàn tay chính là "vũ khí" giúp pose-based **bất biến với người ký và khoảng cách camera** — rất phù hợp với bài toán signer-independent.

### 2.2. Kiến trúc SPOTER (`src/models/spoter/modelling.py`)

SPOTER (Sign POse-based TransformER) là một **Transformer encoder–decoder** rất gọn:

- **Đầu vào**: tensor `(B, T=70, 54, 2)` → flatten thành `(B, 70, 108)`. Lưu ý `hidden_dim = 108 = 54 × 2`, tức **chiều ẩn bị ràng buộc bằng đúng số khớp × số kênh** (không phải siêu tham số tự do).
- **Positional embedding** học được (`row_embed`, `pos`) cộng vào chuỗi frame.
- **Encoder**: 6 lớp Transformer, 9 đầu attention → học quan hệ **không gian–thời gian giữa các frame**.
- **Decoder tuỳ biến** (`SPOTERTransformerDecoderLayer`): **bỏ self-attention**, chỉ giữ cross-attention. Một **"class query" học được** (1 token) đóng vai trò truy vấn toàn bộ chuỗi để tổng hợp thành 1 vector phân loại.
- **Head**: `Linear(108 → 400)`, loss = cross-entropy.

Đây là mô hình **cực nhẹ** (chỉ vài triệu tham số), huấn luyện được với batch 256, vài giờ là xong 350 epoch.

### 2.3. SL-GCN & DSTA-SLR (cùng nhánh, mạnh hơn SPOTER)

Repo còn hỗ trợ hai kiến trúc pose **hiện đại hơn**:

- **SL-GCN**: Graph Convolutional Network coi khung xương là **đồ thị** (khớp = đỉnh, xương = cạnh), tích chập không gian–thời gian trên đồ thị. Thường vượt SPOTER trên từ vựng lớn.
- **DSTA-SLR**: Decoupled Spatial-Temporal Attention — tách riêng attention không gian và thời gian, hỗ trợ thêm luồng **bone stream / motion stream** (vận tốc khớp).

### 2.4. Kết quả huấn luyện SPOTER (run `spoter_v3.0_multicam`)

Cấu hình: `arch=spoter`, modality `pose`, subset `cam_1_2_3` (gộp 3 camera), 70 frame, hidden_dim 108, 350 epoch, batch 256, lr 1e-3, weight_decay 0.05, augmentation bật (aug_prob 0.7 + Gaussian noise). Best checkpoint theo `accuracy` trên val là `checkpoint-75264`.

| Metric | Validation | Test (signer `024`, *unseen*) |
|---|---|---|
| Top-1 accuracy | **93.7%** | **85.7%** |
| Top-5 accuracy | 99.1% | 95.4% |
| Top-10 accuracy | 99.5% | 96.9% |
| F1 | 94.1% | 86.2% |
| Precision | 94.6% | 87.9% |
| Recall | 94.1% | 86.2% |
| Loss | 0.387 | 0.942 |

> So với baseline đơn camera trước đây (test ~81.8% top-1 / 93.8% top-5), bản **đa camera `cam_1_2_3` cải thiện rõ rệt** lên 85.7% / 95.4%. Khoảng cách val→test (~8 điểm) chính là chi phí của thiết lập signer-independent đã nêu ở mục 1.

---

## 3. Nhánh RGB-based: VideoMAE

### 3.1. Pipeline đặc trưng (`src/features/utils.py:get_rgb_transforms`)

1. Giải mã video, **lấy mẫu đều 16 frame** (UniformTemporalSubsample, sample_rate=4).
2. Chuẩn hoá [0,1], resize cạnh ngắn 256–320, **crop 224×224**.
3. Augmentation: lật ngang 0.5, **AugMix** (magnitude 3) — biến đổi cường độ ảnh.

### 3.2. Kiến trúc VideoMAE (`src/models/videomae/modelling.py`)

VideoMAE = **Vision Transformer cho video**, dùng trọng số `MCG-NJU/videomae-small-finetuned-kinetics`:

- **Đầu vào**: `(B, 16, 3, 224, 224)`.
- Video được chia thành **tubelet** (khối 2×16×16 pixel) → mỗi tubelet là 1 token; self-attention toàn cục trên toàn bộ token không gian–thời gian.
- **Tiền huấn luyện** theo kiểu Masked Autoencoder (che ~90% tubelet rồi tái tạo) — học biểu diễn chuyển động mạnh; ở đây dùng bản đã fine-tune phân loại trên Kinetics rồi **fine-tune tiếp toàn bộ** cho VSL (num_frozen_layers=0).
- ~22 triệu tham số (bản small), nặng hơn SPOTER nhiều lần.

> Code chỉ là một **lớp bọc mỏng** quanh `VideoMAEForVideoClassification` của HuggingFace, thay head phân loại bằng 400 lớp.

### 3.3. Các kiến trúc khác trong nhánh RGB (cùng nhánh với VideoMAE)

Nhánh RGB trong repo (`RGB_BASED_MODELS` trong `src/utils/constants.py`) gồm **hai nguồn**: một mô hình từ HuggingFace (VideoMAE) và một loạt backbone video kinh điển nạp từ **TorchHub** (`TORCHHUB_RGB_BASED_MODELS`). Tất cả dùng chung cùng một pipeline đặc trưng RGB (mục 3.1) và đều có thư mục mô hình riêng trong `src/models/` (`videomae`, `swin3d`, `video_resnet`, `s3d`, `mvit`); chỉ cần đổi `model.arch` trong YAML là chuyển mô hình. Bảng dưới liệt kê **toàn bộ `arch` hợp lệ** của nhánh này:

| `arch` | Nguồn | Họ kiến trúc | Ý tưởng cốt lõi | Thư mục / loader |
|---|---|---|---|---|
| `videomae` | HuggingFace | ViT cho video (masked autoencoder) | Tubelet token + self-attention toàn cục; pretrain MAE rồi fine-tune | `src/models/videomae/` |
| `swin3d_t` / `swin3d_s` / `swin3d_b` | TorchHub | **Video Swin Transformer** | Self-attention theo **cửa sổ 3D dịch chuyển** (shifted window), phân cấp đa tỉ lệ — gọn hơn ViT toàn cục | `src/models/swin3d/` |
| `r3d_18` | TorchHub | 3D ResNet | Conv 3D thuần (3×3×3) trên khối ResNet-18 | `src/models/video_resnet/` |
| `mc3_18` | TorchHub | Mixed Conv ResNet | Conv 3D ở tầng đầu, **2D ở tầng sau** — lai 2D/3D giảm chi phí | `src/models/video_resnet/` |
| `r2plus1d_18` | TorchHub | R(2+1)D ResNet | **Tách** conv không gian (2D) khỏi conv thời gian (1D) → nhiều phi tuyến hơn, dễ tối ưu hơn conv 3D đầy đủ | `src/models/video_resnet/` |
| `s3d` | TorchHub | **Separable 3D** (S3D) | Tách conv 3D thành (không gian 2D + thời gian 1D) trên backbone kiểu Inception → nhẹ, hiệu quả | `src/models/s3d/` |
| `mvit_v1_b` / `mvit_v2_s` | TorchHub | **Multiscale Vision Transformer** | ViT **phân cấp đa tỉ lệ**: giảm dần độ phân giải, tăng dần số kênh; v2 thêm decomposed relative position + residual pooling | `src/models/mvit/` |

Phân nhóm theo triết lý:

- **Transformer video**: `videomae`, `swin3d_*`, `mvit_*` — biểu diễn mạnh, trần độ chính xác cao, nhưng nặng và đói dữ liệu.
- **3D CNN**: `r3d_18`, `mc3_18`, `r2plus1d_18`, `s3d` — nhẹ hơn, inductive bias không gian–thời gian mạnh hơn, train ổn định hơn trên dữ liệu vừa phải. `r2plus1d_18` và `s3d` (tách không gian/thời gian) thường là điểm cân bằng tốt giữa chi phí và độ chính xác trong nhóm này.

> Lưu ý kỹ thuật (đọc từ `src/tools/models.py:load_rgb_model_for_training`): các backbone TorchHub được bọc bằng các lớp `*ForVideoClassification` tương ứng (`Swin3DForVideoClassification`, `S3DForVideoClassification`, `MViTForVideoClassification`, và họ video-resnet), thay head phân loại bằng 400 lớp — y hệt cách VideoMAE được bọc. Hiện trong `configs/` mới chỉ có sẵn YAML cho `videomae_s`; các `arch` RGB còn lại đã có code nhưng **cần tự tạo YAML config** để train.

---

## 4. So sánh trực tiếp: SPOTER (Pose) vs VideoMAE (RGB)

| Tiêu chí | **SPOTER (Pose)** | **VideoMAE (RGB)** |
|---|---|---|
| Đầu vào | 54 khớp × (x,y) | 16 frame ảnh 224² |
| Kích thước | Rất nhỏ (~vài triệu) | Lớn (~22M) |
| Tốc độ train | Vài giờ, batch 256 | Chậm, batch 8 |
| Tốc độ inference | Real-time, chạy được CPU/edge | Nặng, cần GPU |
| Bất biến người ký/nền/ánh sáng | **Rất tốt** (đã trừu tượng hoá khỏi điểm ảnh) | **Kém hơn** (nhạy với ngoại hình, quần áo, nền) |
| Bắt chi tiết bàn tay/biểu cảm | Hạn chế (mất chi tiết hình dạng tay, tiếp xúc, mặt) | **Tốt** (giữ toàn bộ thông tin thị giác) |
| Phụ thuộc bên ngoài | **Phụ thuộc nặng vào MediaPipe** (lỗi pose → lỗi lan truyền; nhoè/che khuất tay là điểm yếu chí mạng) | Tự lực, end-to-end |
| Hiệu quả dữ liệu | Cao (ít dữ liệu vẫn học tốt) | Thấp (cần nhiều dữ liệu) |
| Trần độ chính xác (ceiling) | Trung bình | **Cao hơn** nếu đủ dữ liệu |
| Hiện trạng (run `spoter_v3.0_multicam`, đa camera) | test **85.7%** top-1 / **95.4%** top-5 / **96.9%** top-10 (val **93.7%** / top-5 **99.1%**) | Chưa huấn luyện đúng |
| Riêng tư | Tốt (không lưu mặt/ảnh) | Lưu ảnh thô |

### Điểm mấu chốt cho **đúng bài toán này**

1. **Bài toán đo signer-independent** → pose-based có lợi thế cấu trúc: nó *vứt bỏ* ngoại hình người ký, vốn là nguồn gây sai khác lớn nhất. VideoMAE phải *học cách bỏ qua* ngoại hình từ dữ liệu → cần nhiều người ký + augmentation mạnh, nếu không sẽ tổng quát hoá kém hơn sang người mới.
2. **Mục tiêu real-time** (pipeline inference đã dùng MediaPipe Holistic để phát hiện tay lên/xuống) → khung xương **vốn đã có sẵn** trong luồng suy luận. Dùng pose-based gần như miễn phí; dùng VideoMAE phải thêm một backbone nặng.
3. **400 lớp với nhiều ký hiệu khác nhau ở hình dạng bàn tay tinh tế** → đây là điểm RGB thắng: SPOTER chỉ có 21 điểm/tay, dễ nhầm các ký hiệu khác nhau ở độ cong ngón tay/tiếp xúc mà MediaPipe không bắt chính xác.

---

## 5. Kết luận & khuyến nghị (góc nhìn nghiên cứu)

**Phương pháp phù hợp nhất *về tổng thể* cho dự án: nhánh Pose-based** — vì nó khớp với (a) thước đo signer-independent, (b) yêu cầu real-time, (c) hiệu quả dữ liệu, (d) MediaPipe đã có sẵn trong luồng. Đây nên là **mô hình chủ lực để triển khai**.

**Nhưng SPOTER không phải là lựa chọn tối ưu *trong* nhánh pose.** SPOTER là kiến trúc khá cũ (2021) và đơn giản; với từ vựng 400 lớp, **SL-GCN hoặc DSTA-SLR thường cho độ chính xác cao hơn rõ rệt** vì khai thác được cấu trúc đồ thị xương và luồng vận động. Khuyến nghị **thử SL-GCN/DSTA-SLR** sau khi đã trích xuất xong pose 3 camera — chi phí gần như bằng 0 vì dữ liệu pose đã sẵn sàng.

**VideoMAE *không nên* là mô hình triển khai chính**, vì:
- Nặng, khó real-time/edge.
- Nhạy với người ký mới → rủi ro tổng quát hoá kém hơn trên đúng tập test của dự án.

**Tuy nhiên VideoMAE rất đáng giá ở hai vai trò:**
1. **Mô hình "trần trên"** (accuracy ceiling) để biết giới hạn của RGB.
2. **Bổ trợ cho pose qua ensemble/fusion**: kết hợp pose (bất biến người ký) + RGB (chi tiết bàn tay) thường cho kết quả **tốt nhất** trong các nghiên cứu SLR. Đây là hướng cho độ chính xác cao nhất nếu tài nguyên cho phép.

### Lộ trình đề xuất (theo thứ tự ưu tiên)

1. **Ngay bây giờ**: hoàn tất trích xuất pose 3 camera → train SPOTER `cam_1_2_3` để có baseline đa góc.
2. **Tiếp theo**: train **SL-GCN/DSTA-SLR** trên cùng dữ liệu pose — nhiều khả năng vượt SPOTER, gần như miễn phí.
3. **Song song/cuối**: train **VideoMAE** đa camera để đo trần RGB.
4. **Nâng cao** (nếu cần đẩy SOTA): **fusion pose + RGB**.

> **Tóm gọn một câu:** Pose-based là "đúng" cho triển khai và cho đúng thước đo của dự án; trong đó nên nâng cấp từ SPOTER lên SL-GCN/DSTA-SLR. VideoMAE là mô hình mạnh để tham chiếu trần và để fusion, chứ không phải lựa chọn triển khai chính.

---

## 6. Tìm hiểu sâu: SL-GCN và DSTA-SLR

Cả hai đều thuộc **nhánh pose** (giống SPOTER về đầu vào là khung xương), nhưng khác nhau căn bản về **cách mô hình hoá cấu trúc không gian–thời gian của bộ xương**. Đây là điểm cốt lõi.

### 6.1. Khác biệt nền tảng nhất: cách biểu diễn bộ xương

| | **SPOTER** | **SL-GCN / DSTA-SLR** |
|---|---|---|
| Số khớp (V) | 54 | 27 (`SLGCN_JOINTS`) |
| Số kênh (C) | **2** (x, y) | **3** (x, y, **confidence**) |
| Cách đưa vào mạng | Mỗi frame bị **làm phẳng** thành 1 vector 108-chiều → mạng chỉ thấy một *chuỗi token theo thời gian*, **mất cấu trúc đồ thị khớp** | Giữ nguyên tensor có cấu trúc `(C, T, V, M)` → mạng mô hình hoá **tường minh quan hệ giữa từng khớp** (không gian) lẫn theo thời gian |
| Tổng hợp đặc trưng | "class query" token học được (decoder) | Global Average Pooling trên (T, V) |

Ý nghĩa: SPOTER coi một frame như "một túi toạ độ" và chỉ học quan hệ *giữa các frame*. Ngược lại, SL-GCN/DSTA giữ lại **topology của bộ xương** và học quan hệ *giữa các khớp với nhau* — đúng bản chất của ngôn ngữ ký hiệu (ngón tay với cổ tay, tay với vai…). Ngoài ra, việc giữ **kênh confidence** giúp SL-GCN/DSTA có thể "giảm trọng số" những khớp mà MediaPipe dự đoán không chắc chắn — một lợi thế chống nhiễu mà SPOTER không có (vì đã vứt confidence).

### 6.2. SL-GCN — tiếp cận bằng tích chập đồ thị (`src/models/sl_gcn/modelling.py`)

Đây là biến thể **ST-GCN** (Spatial-Temporal Graph Convolutional Network):

- **Đầu vào**: `(N, C=3, T, V=27, M=1)` (M = số người).
- **Khối ST-GCN** (`STGCNBlock`) gồm 2 thành phần xen kẽ:
  - `SpatialGraphConvolution`: tích chập đồ thị — tổng hợp đặc trưng từ các **khớp lân cận** dựa trên **ma trận kề (adjacency)**, thực hiện bằng `einsum("nctv,vw->nctw", x, A)`.
  - `TemporalConvolution`: conv 1D (kernel 9) dọc **trục thời gian** cho từng khớp.
  - Có **kết nối tắt (residual)**.
- **Thân mạng**: 9 khối ST-GCN, số kênh tăng dần `64 → 128 → 256`, giảm độ phân giải thời gian (stride=2) ở khối 4 và 7. Sau đó **global average pooling** → `Linear(256 → 400)`.
- **Đa luồng (multi-stream)**: hỗ trợ `bone_stream` (vector xương = hiệu toạ độ giữa 2 khớp nối) và `motion_stream` (vận tốc = hiệu toạ độ giữa 2 frame liên tiếp). Đây là kỹ thuật **ensemble nhiều luồng** kinh điển của họ GCN, thường tăng đáng kể độ chính xác — SPOTER hoàn toàn không có.

> **Lưu ý kỹ thuật quan trọng (trung thực):** Hàm `get_adjacency_matrix` trong repo hiện chỉ dựng **đồ thị dạng chuỗi** (`adjacency[i, i+1] = 1`, nối các khớp liên tiếp theo chỉ số) — *không phải* đồ thị giải phẫu thật của bàn tay/cơ thể. Docstring tự ghi là "simple/baseline". Bản SL-GCN gốc (Decoupled GCN + DropGraph + đồ thị tay/thân chuẩn) mạnh hơn nhiều. Vì vậy, để khai thác hết sức mạnh SL-GCN, **nên thay ma trận kề bằng đồ thị xương đúng** cho 27 khớp.

### 6.3. DSTA-SLR — tiếp cận bằng attention tách rời (`src/models/dsta_slr/modelling.py`)

DSTA = **Decoupled Spatial-Temporal Attention**. Thay vì tích chập, mô hình này dùng **self-attention** trên bộ xương, và "tách rời" (decoupled) không gian khỏi thời gian:

- **Đầu vào**: `(N, C=3, T, V=27, M=1)` → `Linear(3 → inner_dim=64)`.
- **Khối DSTA** (`DSTABlock`), lặp `depth=4` lần, mỗi khối gồm:
  - `SpatialAttention`: self-attention **giữa 27 khớp** *tại mỗi thời điểm* (ma trận attention `V×V`) → học khớp nào tương tác với khớp nào.
  - `TemporalAttention`: self-attention **dọc thời gian** *cho từng khớp riêng* (ma trận attention `T×T`) → học động lực thời gian.
  - `FFN` (mở rộng ×4) + LayerNorm + residual (đúng kiểu Transformer block).
- **Kết thúc**: LayerNorm → global average pooling trên (T, V) → `Linear(64 → 400)`.
- Cũng hỗ trợ `bone_stream` / `motion_stream`.

**"Decoupled" nghĩa là gì?** Nếu làm attention không-thời-gian *đồng thời* trên toàn bộ `T×V` token thì chi phí là `O((T·V)²)` — rất tốn. DSTA **tách** thành: attention không gian (`O(V²)`) rồi attention thời gian (`O(T²)`) → rẻ hơn nhiều mà vẫn nắm được cả hai chiều. Khác với GCN (locality cứng theo đồ thị), attention cho phép **mọi khớp "nhìn thấy" mọi khớp khác** (quan hệ tầm xa, ví dụ hai bàn tay phối hợp) — linh hoạt hơn nhưng cần nhiều dữ liệu hơn.

### 6.4. So sánh tổng hợp 3 mô hình pose

| Tiêu chí | **SPOTER** | **SL-GCN** | **DSTA-SLR** |
|---|---|---|---|
| Ý tưởng cốt lõi | Transformer trên *chuỗi frame phẳng* | Tích chập trên *đồ thị xương* | Attention *tách rời* không gian/thời gian |
| Mô hình hoá quan hệ giữa các khớp | **Ngầm** (đã làm phẳng) | **Tường minh** (đồ thị, locality) | **Tường minh** (attention, toàn cục) |
| Số kênh / dùng confidence | 2, không | 3, **có** | 3, **có** |
| Đa luồng bone/motion | Không | **Có** | **Có** |
| Inductive bias | Yếu về không gian | Mạnh (locality đồ thị) → hiệu quả dữ liệu | Trung bình (linh hoạt, cần nhiều dữ liệu) |
| Cách tổng hợp | class-query token | Global avg pool | Global avg pool |
| Độ phức tạp tính toán | Thấp | Thấp–trung bình | Trung bình |
| Kỳ vọng độ chính xác (400 lớp) | Cơ sở | Thường **cao hơn SPOTER** | Thường **cao hơn SPOTER**, mạnh nếu đủ dữ liệu |

**Tóm tắt khác biệt với SPOTER:** SPOTER bỏ qua cấu trúc đồ thị của bộ xương (làm phẳng mỗi frame) và chỉ dùng Transformer theo thời gian. SL-GCN và DSTA-SLR **giữ và khai thác tường minh topology khớp** (qua graph-conv hoặc spatial-attention), **dùng thêm kênh confidence** và **hỗ trợ đa luồng bone/motion** — đó là lý do chúng thường vượt SPOTER trên từ vựng lớn.

### 6.5. Tình trạng trong repo: đã được thêm chưa?

**Có — đã được hiện thực và "đấu dây" đầy đủ trong code, nhưng CHƯA có file config sẵn để chạy.**

✅ Đã có:
- Thư mục mô hình đầy đủ: `src/models/sl_gcn/` và `src/models/dsta_slr/` (đủ `configuration.py`, `modelling.py`, `__init__.py`).
- Được export trong `src/models/__init__.py` và đăng ký trong `POSE_BASED_MODELS` (`src/utils/constants.py`).
- `src/tools/models.py:load_pose_model_for_training` đã có nhánh dispatch cho `sl_gcn` và `dsta_slr`.
- Pipeline đặc trưng đã có: `src/features/transforms/sl_gcn.py` (chọn 27 khớp, pad, bone/motion stream, normalize) và `src/features/augmentations/sl_gcn.py`.
- Pipeline inference cũng đã hỗ trợ (`SLGCNGraphClassificationPipeline` trong `src/pipelines/`).

❌ Chưa có (việc cần làm để train):
- **Không có YAML config** cho chúng. Trong `configs/training/`, `configs/evaluation/`, `configs/inference/` mới chỉ có `spoter*` và `videomae_s`. → Cần tạo `configs/training/sl_gcn.yaml` và `configs/training/dsta_slr.yaml`.
- Như mục 6.2, **ma trận kề của SL-GCN đang là đồ thị chuỗi đơn giản hoá**, nên thay bằng đồ thị xương thật cho 27 khớp để đạt hiệu năng tối đa.

> **Kết luận mục 6:** Mã nguồn cho SL-GCN/DSTA-SLR đã sẵn sàng — chỉ cần thêm config (và lý tưởng là sửa lại đồ thị kề của SL-GCN) là có thể huấn luyện ngay trên dữ liệu pose vừa trích xuất, gần như **không tốn thêm chi phí dữ liệu**. Đây là hướng nâng cấp đáng giá nhất sau SPOTER.
