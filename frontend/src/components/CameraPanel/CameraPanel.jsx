import { useRef, useCallback } from 'react'
import useCamera from '../../hooks/useCamera'
import useSignStore from '../../store/useSignStore'
import VideoDisplay from './VideoDisplay'
import ManualPanel from '../ManualPanel/ManualPanel'
import wsClient from '../../services/ws'

const IDLE_FINALIZE_MS = 2500 // Nghỉ tay 2.5s -> tự động ghép câu + đọc

export default function CameraPanel({ activeTab }) {
  const videoRef = useRef(null)
  const isCameraOn = useSignStore(s => s.isCameraOn)
  const { error } = useCamera(videoRef, isCameraOn)

  // MediaPipe Holistic (trong VideoDisplay) gọi lại mỗi frame với keypoint đã trích
  // (SPOTER 108 số + SL-GCN 81 số cho late-fusion).
  const handleKeypoints = useCallback((keypoints, slgcnKeypoints) => {
    wsClient.sendKeypoints(keypoints, slgcnKeypoints)
    if (wsClient.shouldFinalize(IDLE_FINALIZE_MS)) {
      wsClient.sendKeypoints(keypoints, slgcnKeypoints, true)
    }
  }, [])

  return (
    <div className="flex flex-col h-full min-h-0">
      {/* Dynamic Content: Realtime video display or Manual Upload panel */}
      {activeTab === 'manual' ? (
        <ManualPanel />
      ) : (
        <VideoDisplay videoRef={videoRef} error={error} onKeypoints={handleKeypoints} />
      )}
    </div>
  )
}


