const _urls = new Set()

/**
 * Converts base64 audio string (from WS message audio_b64) to a blob URL.
 * This Blob URL can be assigned directly to an <audio> element.
 */
export const base64ToAudioUrl = (base64, mimeType = 'audio/wav') => {
  const binaryString = atob(base64)
  const len = binaryString.length
  const bytes = new Uint8Array(len)
  for (let i = 0; i < len; i++) {
    bytes[i] = binaryString.charCodeAt(i)
  }
  const blob = new Blob([bytes], { type: mimeType })
  const url = URL.createObjectURL(blob)
  _urls.add(url)
  return url
}

/**
 * Tạo blob URL từ khung nhị phân WAV (backend gửi qua WebSocket).
 */
export const blobToAudioUrl = (blob) => {
  const url = URL.createObjectURL(blob)
  _urls.add(url)
  return url
}

export const revokeAudioUrl = (url) => {
  if (url?.startsWith('blob:')) {
    URL.revokeObjectURL(url)
    _urls.delete(url)
  }
}

// Clean up all blob URLs when page unloads
window.addEventListener('beforeunload', () => {
  _urls.forEach(url => URL.revokeObjectURL(url))
})
