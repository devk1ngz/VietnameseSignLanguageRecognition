const BASE_URL = import.meta.env.VITE_API_URL

const request = async (path, options = {}) => {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  if (!res.ok) throw new Error(`API error ${res.status}: ${path}`)
  return res
}

export const api = {
  // Nhận dạng MỘT video (upload/quay) -> { glosses: string[] }. Dùng multipart:
  // KHÔNG tự set Content-Type để trình duyệt tự thêm boundary.
  recognizeVideo: async (file) => {
    const form = new FormData()
    form.append('file', file)
    const res = await fetch(`${BASE_URL}/api/recognize-video`, {
      method: 'POST',
      body: form,
    })
    if (!res.ok) throw new Error(`API error ${res.status}: /api/recognize-video`)
    return res.json() // { glosses }
  },

  // Ghép danh sách từ -> câu tiếng Việt (LLM) + giọng nói (TTS).
  composeSpeech: async (glosses) => {
    const res = await request('/api/compose', {
      method: 'POST',
      body: JSON.stringify({ glosses }),
    })
    return res.json() // { sentence, audio_b64, mime }
  },
}
