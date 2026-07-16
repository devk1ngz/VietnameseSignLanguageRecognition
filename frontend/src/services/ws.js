import { blobToAudioUrl } from '../utils/audioHelper'

const WS_URL = import.meta.env.VITE_WS_URL

class WebSocketClient {
  constructor() {
    this.ws = null
    this.handlers = {}
    this.reconnectTimer = null
    this.shouldReconnect = false
    this.reconnectAttempts = 0
    // Trạng thái để tự động chốt câu khi nghỉ tay (idle finalize).
    this.lastGlossAt = 0
    this.hasPending = false // có gloss chưa được ghép câu
  }

  connect({ onStatus, onGloss, onSentence, onAudio, onProcessing }) {
    this.handlers = { onStatus, onGloss, onSentence, onAudio, onProcessing }
    this.shouldReconnect = true
    this.reconnectAttempts = 0 // Reset attempts on fresh connect
    this._open()
  }

  _open() {
    if (this.reconnectAttempts >= 5) {
      console.warn(`WebSocket connection failed after 5 attempts. Stopping reconnect.`)
      this.shouldReconnect = false
      this.handlers.onStatus?.('disconnected')
      return
    }

    this.reconnectAttempts++
    this.handlers.onStatus?.('connecting')

    try {
      this.ws = new WebSocket(WS_URL)
    } catch (e) {
      console.error('WebSocket connection initialization failed:', e)
      this.handlers.onStatus?.('error')
      this._handleReconnect()
      return
    }

    this.ws.onopen = () => {
      this.reconnectAttempts = 0 // Reset attempts on successful connection
      this.handlers.onStatus?.('connected')
    }

    this.ws.onclose = () => {
      // If we intentionally stopped reconnecting, don't set status to error or reconnect
      if (this.shouldReconnect) {
        this.handlers.onStatus?.('disconnected')
        this._handleReconnect()
      }
    }

    this.ws.onerror = () => {
      if (this.shouldReconnect) {
        this.handlers.onStatus?.('error')
      }
    }

    this.ws.onmessage = (event) => {
      // Khung nhị phân = WAV từ TTS.
      if (event.data instanceof Blob) {
        const blobUrl = blobToAudioUrl(event.data)
        this.handlers.onAudio?.(blobUrl)
        this.handlers.onProcessing?.(false)
        this.hasPending = false
        return
      }

      try {
        const data = JSON.parse(event.data)
        switch (data.type) {
          case 'gloss':
            this.lastGlossAt = Date.now()
            this.hasPending = true
            this.handlers.onGloss?.(data.gloss)
            break
          case 'processing':
            this.handlers.onProcessing?.(true)
            break
          case 'sentence':
            this.handlers.onSentence?.(data.text)
            break
          case 'error':
            console.error('Backend error:', data.error)
            this.handlers.onProcessing?.(false)
            break
          default:
            console.warn('Unknown message type received:', data.type)
        }
      } catch (err) {
        console.error('Error parsing WS message:', err)
      }
    }
  }

  _handleReconnect() {
    if (this.shouldReconnect) {
      clearTimeout(this.reconnectTimer)
      this.reconnectTimer = setTimeout(() => this._open(), 3000)
    }
  }

  /**
   * Gửi một frame keypoint. endSign=true để backend ghép câu + đọc.
   * slgcnKeypoints (tuỳ chọn, 81 số) cho late-fusion SL-GCN; thiếu -> backend chạy SPOTER-only.
   */
  sendKeypoints(keypoints, slgcnKeypoints = null, endSign = false) {
    if (this.ws?.readyState !== WebSocket.OPEN) return
    const payload = { keypoints, end_sign: endSign }
    if (slgcnKeypoints) payload.slgcn_keypoints = slgcnKeypoints
    this.ws.send(JSON.stringify(payload))
    if (endSign) this.hasPending = false
  }

  /**
   * True khi có gloss mới và đã nghỉ tay đủ lâu (idleMs) -> nên chốt câu.
   */
  shouldFinalize(idleMs) {
    return this.hasPending && Date.now() - this.lastGlossAt >= idleMs
  }

  disconnect() {
    this.shouldReconnect = false
    clearTimeout(this.reconnectTimer)
    if (this.ws) {
      this.ws.close()
      this.ws = null
    }
  }
}

export default new WebSocketClient()
