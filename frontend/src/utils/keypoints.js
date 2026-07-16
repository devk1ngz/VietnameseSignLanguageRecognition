/**
 * Chuyển kết quả MediaPipe Holistic thành vector 108 số (54 khớp x (x, y))
 * đúng định dạng SPOTER backend mong đợi.
 *
 * Thứ tự khớp = thứ tự JointSelect của pipeline (pose_format):
 *   12 khớp thân, rồi 21 khớp tay TRÁI, rồi 21 khớp tay PHẢI.
 * Toạ độ là PIXEL (MediaPipe trả 0..1 nên nhân với width/height).
 * Khớp neck (index 1) luôn = [0, 0]; khớp thiếu (không nhận diện) = [0, 0].
 */

// Chỉ số landmark MediaPipe Pose theo thứ tự thân của JointSelect (null = neck synthesize 0).
const POSE_MAP = [0, null, 5, 2, 8, 7, 12, 11, 14, 13, 16, 15]

// Chỉ số landmark MediaPipe Hand theo thứ tự tay của JointSelect (dùng cho cả 2 tay).
const HAND_MAP = [0, 8, 7, 6, 5, 12, 11, 10, 9, 16, 15, 14, 13, 19, 20, 18, 17, 4, 3, 2, 1]

const KEYPOINTS_LEN = (POSE_MAP.length + HAND_MAP.length * 2) * 2 // 108

export function extractKeypoints(results, width, height) {
  const out = new Array(KEYPOINTS_LEN).fill(0)
  let o = 0

  const pose = results.poseLandmarks
  for (const idx of POSE_MAP) {
    const lm = idx !== null && pose ? pose[idx] : null
    if (lm) {
      out[o] = lm.x * width
      out[o + 1] = lm.y * height
    }
    o += 2
  }

  for (const hand of [results.leftHandLandmarks, results.rightHandLandmarks]) {
    for (const idx of HAND_MAP) {
      const lm = hand ? hand[idx] : null
      if (lm) {
        out[o] = lm.x * width
        out[o + 1] = lm.y * height
      }
      o += 2
    }
  }

  return out
}

// ===== SL-GCN keypoints (nhánh 2 của late-fusion) =====
// 27 khớp x (x, y, confidence) = 81 số, thứ tự SLGCN_JOINTS[27]: 7 khớp thân, 10 tay TRÁI, 10 tay PHẢI.
// Tái hiện ĐÚNG cách pose_format gán confidence: thân = visibility của MediaPipe Pose,
// tay = 1.0 nếu bàn tay được phát hiện (cả 10 khớp), ngược lại 0. Khớp thiếu = [0, 0, 0].
// Server chuẩn hoá (normalize_distribution) từ các số này — đã kiểm chứng khớp bit-exact với AI core.

// Chỉ số landmark MediaPipe Pose cho 7 khớp thân SL-GCN:
//   nose, leftShoulder, rightShoulder, leftElbow, rightElbow, leftWrist, rightWrist
const SLGCN_POSE_MAP = [0, 11, 12, 13, 14, 15, 16]

// Chỉ số landmark MediaPipe Hand cho 10 khớp tay SL-GCN (dùng cho cả 2 tay):
//   wrist, thumbTip, indexMCP, indexTip, middleMCP, middleTip, ringMCP, ringTip, pinkyMCP, pinkyTip
const SLGCN_HAND_MAP = [0, 4, 5, 8, 9, 12, 13, 16, 17, 20]

const SLGCN_KEYPOINTS_LEN = (SLGCN_POSE_MAP.length + SLGCN_HAND_MAP.length * 2) * 3 // 81

export function extractSlgcnKeypoints(results, width, height) {
  const out = new Array(SLGCN_KEYPOINTS_LEN).fill(0)
  let o = 0

  const pose = results.poseLandmarks
  for (const idx of SLGCN_POSE_MAP) {
    const lm = pose ? pose[idx] : null
    if (lm) {
      out[o] = lm.x * width
      out[o + 1] = lm.y * height
      out[o + 2] = lm.visibility ?? 0 // confidence thân = visibility
    }
    o += 3
  }

  // Tay TRÁI rồi tay PHẢI (khớp thứ tự LEFT/RIGHT_HAND_LANDMARKS của pose_format).
  for (const hand of [results.leftHandLandmarks, results.rightHandLandmarks]) {
    for (const idx of SLGCN_HAND_MAP) {
      const lm = hand ? hand[idx] : null
      if (lm) {
        out[o] = lm.x * width
        out[o + 1] = lm.y * height
        out[o + 2] = 1.0 // confidence tay = 1.0 khi bàn tay được phát hiện
      }
      o += 3
    }
  }

  return out
}
