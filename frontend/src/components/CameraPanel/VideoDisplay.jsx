import { useRef, useEffect, useState } from 'react'
import { Camera, CameraOff, Activity, Loader2 } from 'lucide-react'
import useSignStore from '../../store/useSignStore'
import { extractKeypoints, extractSlgcnKeypoints } from '../../utils/keypoints'

// Singleton instance to prevent multiple asset downloads and re-initialization
let globalHolistic = null
let isHolisticLoading = false
let onResultsCallback = null

export default function VideoDisplay({ videoRef, error, onKeypoints }) {
  const isCameraOn = useSignStore(s => s.isCameraOn)
  const setCameraOn = useSignStore(s => s.setCameraOn)
  const showSkeleton = useSignStore(s => s.showSkeleton)
  const toggleSkeleton = useSignStore(s => s.toggleSkeleton)
  const canvasRef = useRef(null)

  const [isModelLoading, setIsModelLoading] = useState(false)

  // 1. Eagerly warm up/initialize MediaPipe Holistic once when camera is on
  useEffect(() => {
    if (!isCameraOn || globalHolistic || isHolisticLoading) return

    let retryTimer = null
    let cancelled = false

    const initMediaPipe = async () => {
      if (cancelled) return
      if (!window.Holistic) {
        // Retry shortly if scripts are still downloading
        retryTimer = setTimeout(initMediaPipe, 500)
        return
      }

      try {
        isHolisticLoading = true
        setIsModelLoading(true)

        const holistic = new window.Holistic({
          locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`
        })

        // Cân bằng giữa độ trễ và độ chính xác của bàn tay.
        holistic.setOptions({
          // 1 = Full: model pose tốt hơn -> vùng cắt (ROI) bàn tay chuẩn hơn ->
          // KHÔNG bị mất khung xương tay khi đưa tay lên. (0/Lite bắt tay rất kém.)
          // Nếu máy yếu và thấy trễ nặng, hạ lại về 0.
          modelComplexity: 1,
          smoothLandmarks: true,
          refineFaceLandmarks: false, // Không cần face mesh chi tiết cho app này
          // Hạ ngưỡng để bắt được tay khi chuyển động nhanh (đưa tay lên).
          minDetectionConfidence: 0.4,
          minTrackingConfidence: 0.4
        })

        holistic.onResults((results) => {
          if (onResultsCallback) {
            onResultsCallback(results)
          }
        })

        globalHolistic = holistic
        isHolisticLoading = false
        if (!cancelled) setIsModelLoading(false)
        console.log("MediaPipe Holistic warmed up and ready.")
      } catch (err) {
        console.error("Failed to eagerly load MediaPipe Holistic:", err)
        isHolisticLoading = false
        if (!cancelled) setIsModelLoading(false)
      }
    }

    initMediaPipe()

    return () => {
      // Dừng vòng retry chờ script CDN khi tắt camera/unmount.
      cancelled = true
      if (retryTimer) clearTimeout(retryTimer)
    }
  }, [isCameraOn])

  // 2. Chạy Holistic liên tục khi camera bật: trích + gửi keypoint mỗi frame,
  //    vẽ khung xương chỉ khi được bật.
  useEffect(() => {
    const video = videoRef.current
    if (!video) return

    let active = true
    let lastSend = 0
    let sending = false
    // ~25 fps: 70 frame buffer của backend ~ 2.8s, xấp xỉ độ dài một ký hiệu.
    const SEND_INTERVAL_MS = 40

    onResultsCallback = (results) => {
      if (!active) return

      // Gửi keypoint cho backend (độc lập với việc vẽ khung xương).
      const w = video.videoWidth
      const h = video.videoHeight
      if (w && h && onKeypoints) {
        // Gửi CẢ keypoint SPOTER (108) và SL-GCN (81) cho late-fusion phía backend.
        onKeypoints(extractKeypoints(results, w, h), extractSlgcnKeypoints(results, w, h))
      }

      // Vẽ khung xương chỉ khi bật.
      const canvas = canvasRef.current
      const ctx = canvas?.getContext('2d')
      if (!showSkeleton || !canvas || !ctx || !w || !h) return

      // Canvas dùng TOẠ ĐỘ GỐC của video (videoWidth/Height). Canvas có cùng
      // CSS object-cover với <video> nên trình duyệt scale y hệt -> khung xương
      // khớp chính xác với hình (trước đây canvas theo kích thước container -> lệch).
      // Chỉ gán lại size khi đổi (gán canvas.width mỗi frame sẽ cấp phát lại buffer).
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w
        canvas.height = h
      }
      ctx.clearRect(0, 0, canvas.width, canvas.height)

      // Không vẽ face mesh: app chỉ dùng thân + 2 bàn tay, vẽ FACEMESH_TESSELATION
      // (hàng nghìn đoạn) mỗi frame là nguyên nhân chính gây trễ khung hình.
      if (results.poseLandmarks && window.drawConnectors && window.POSE_CONNECTIONS) {
        window.drawConnectors(ctx, results.poseLandmarks, window.POSE_CONNECTIONS, {
          color: '#6366f1',
          lineWidth: 2
        })
        window.drawLandmarks(ctx, results.poseLandmarks, {
          color: '#ffffff',
          fillColor: '#6366f1',
          lineWidth: 1.2,
          radius: 2.5
        })
      }
      if (results.leftHandLandmarks && window.drawConnectors && window.HAND_CONNECTIONS) {
        window.drawConnectors(ctx, results.leftHandLandmarks, window.HAND_CONNECTIONS, {
          color: '#10b981',
          lineWidth: 1.5
        })
        window.drawLandmarks(ctx, results.leftHandLandmarks, {
          color: '#ffffff',
          fillColor: '#10b981',
          lineWidth: 1,
          radius: 2
        })
      }
      if (results.rightHandLandmarks && window.drawConnectors && window.HAND_CONNECTIONS) {
        window.drawConnectors(ctx, results.rightHandLandmarks, window.HAND_CONNECTIONS, {
          color: '#f59e0b',
          lineWidth: 1.5
        })
        window.drawLandmarks(ctx, results.rightHandLandmarks, {
          color: '#ffffff',
          fillColor: '#f59e0b',
          lineWidth: 1,
          radius: 2
        })
      }
    }

    const processLoop = async () => {
      if (!active || !isCameraOn) return
      const now = performance.now()
      if (globalHolistic && video.readyState >= 2 && !sending && now - lastSend >= SEND_INTERVAL_MS) {
        lastSend = now
        sending = true
        try {
          await globalHolistic.send({ image: video })
        } catch {
          // Bỏ qua cảnh báo frame bị skip
        }
        sending = false
      }
      if (active && isCameraOn) requestAnimationFrame(processLoop)
    }

    if (isCameraOn) processLoop()

    return () => {
      active = false
      onResultsCallback = null
    }
  }, [showSkeleton, isCameraOn, videoRef, onKeypoints])

  return (
    <div className="flex-1 relative bg-slate-100 dark:bg-slate-950 flex items-center justify-center overflow-hidden transition-colors duration-300 group">
      
      {/* Mirror effect video - always kept in DOM to avoid play promise abort errors */}
      <video
        ref={videoRef}
        className={`w-full h-full object-cover scale-x-[-1] transition-all duration-500 ${
          isCameraOn && !error && videoRef.current?.srcObject ? 'block opacity-100' : 'hidden opacity-0'
        }`}
        muted
        playsInline
      />

      {/* MediaPipe skeleton canvas overlay */}
      {isCameraOn && showSkeleton && !error && (
        <canvas
          ref={canvasRef}
          className="absolute inset-0 w-full h-full object-cover z-15 pointer-events-none scale-x-[-1]"
        />
      )}

      {/* MediaPipe Warmed Up / Loading Indicator */}
      {isCameraOn && (isModelLoading || (showSkeleton && !globalHolistic)) && !error && (
        <div className="absolute top-16 right-4 z-20 flex items-center gap-2 bg-white/80 dark:bg-slate-950/80 backdrop-blur-md px-3 py-1.5 rounded-full border border-slate-200 dark:border-slate-800 text-[10px] font-semibold text-slate-700 dark:text-slate-300 uppercase tracking-wider shadow-sm">
          <Loader2 size={12} className="animate-spin text-indigo-500 dark:text-indigo-400" />
          <span>Đang tải bộ xương AI...</span>
        </div>
      )}

      {/* Camera disabled screen */}
      {!isCameraOn && (
        <div className="flex flex-col items-center gap-4 text-center z-10 max-w-sm px-6">
          <div className="p-5 rounded-3xl bg-slate-200 dark:bg-slate-900 border border-slate-300 dark:border-slate-800 shadow-xl text-slate-400 dark:text-slate-500">
            <CameraOff size={36} />
          </div>
          <div>
            <h3 className="text-sm font-semibold text-slate-700 dark:text-slate-300">Camera đang tắt</h3>
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">Bật camera để bắt đầu nhận dạng ngôn ngữ ký hiệu.</p>
          </div>
          <button
            onClick={() => setCameraOn(true)}
            className="mt-2 px-5 py-2 rounded-xl bg-indigo-500 hover:bg-indigo-600 text-xs font-semibold text-white shadow-lg shadow-indigo-500/20 transition-all"
          >
            Bật camera
          </button>
        </div>
      )}

      {/* Camera starting placeholder screen */}
      {isCameraOn && !videoRef.current?.srcObject && !error && (
        <div className="flex flex-col items-center gap-3 text-slate-500 dark:text-slate-400 z-10 animate-pulse">
          <div className="p-4 rounded-full bg-slate-200 dark:bg-slate-900 border border-slate-300 dark:border-slate-800">
            <Camera size={32} className="text-slate-600 dark:text-slate-400" />
          </div>
          <p className="text-sm font-medium tracking-wide">Đang kết nối luồng camera...</p>
        </div>
      )}

      {/* Camera permission error / general error */}
      {error && isCameraOn && (
        <div className="flex flex-col items-center gap-4 text-center px-8 text-slate-600 dark:text-slate-400 z-10 max-w-sm">
          <div className="p-4 rounded-full bg-rose-500/10 border border-rose-500/20 text-rose-500 dark:text-rose-400">
            <CameraOff size={32} />
          </div>
          <div>
            <p className="text-sm font-semibold text-rose-700 dark:text-rose-200">Không truy cập được Camera</p>
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-2 leading-relaxed">{error}</p>
          </div>
        </div>
      )}

      {/* Live recording flashing indicator */}
      {isCameraOn && videoRef.current?.srcObject && !error && (
        <div className="absolute top-4 left-4 z-20 flex items-center gap-2 bg-white/80 dark:bg-slate-950/80 backdrop-blur-md px-3 py-1.5 rounded-full border border-slate-200 dark:border-slate-800 text-[10px] font-semibold text-slate-700 dark:text-slate-300 uppercase tracking-wider shadow-sm">
          <span className="relative flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-rose-400 opacity-75"></span>
            <span className="relative inline-flex rounded-full h-2 w-2 bg-rose-500"></span>
          </span>
          <span>LIVE CAM</span>
        </div>
      )}

      {/* Floating Control Bar for Camera toggle and Skeleton Toggle */}
      {isCameraOn && (
        <div className="absolute bottom-6 z-20 flex gap-3">
          <button
            onClick={() => setCameraOn(false)}
            className="flex items-center gap-2 px-4 py-2.5 rounded-full bg-slate-900/90 dark:bg-slate-950/90 hover:bg-slate-800 dark:hover:bg-slate-900 border border-slate-800 dark:border-slate-800 backdrop-blur-md text-xs font-semibold text-rose-400 shadow-xl transition-all duration-200"
          >
            <CameraOff size={14} />
            <span>Tắt Camera</span>
          </button>

          <button
            onClick={toggleSkeleton}
            className={`flex items-center gap-2 px-4 py-2.5 rounded-full border backdrop-blur-md text-xs font-semibold shadow-xl transition-all duration-200 ${
              showSkeleton
                ? 'bg-emerald-500/90 border-emerald-400 text-white hover:bg-emerald-600/90'
                : 'bg-slate-900/90 dark:bg-slate-950/90 border-slate-800 dark:border-slate-800 text-emerald-400 hover:bg-slate-800 dark:hover:bg-slate-900'
            }`}
          >
            <Activity size={14} />
            <span>{showSkeleton ? 'Tắt khung xương' : 'Hiện khung xương'}</span>
          </button>
        </div>
      )}

    </div>
  )
}
