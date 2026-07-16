import { useState, useRef, useEffect } from 'react'
import {
  UploadCloud,
  Trash2,
  Play,
  FileVideo,
  Sparkles,
  CheckCircle,
  Loader2,
  Camera,
  Square,
  RefreshCw,
  Send,
  AlertCircle
} from 'lucide-react'
import useSignStore from '../../store/useSignStore'
import { api } from '../../services/api'
import { base64ToAudioUrl } from '../../utils/audioHelper'

export default function ManualPanel() {
  const [activeTab, setActiveTab] = useState('upload') // 'upload' | 'record'

  // -- Store Actions (đều nhận `source` để tách KQ theo từng tính năng) --
  const clearSession = useSignStore(s => s.clearSession)
  const addGloss = useSignStore(s => s.addGloss)
  const setSentence = useSignStore(s => s.setSentence)
  const setProcessing = useSignStore(s => s.setProcessing)
  const setAudioUrl = useSignStore(s => s.setAudioUrl)
  const setActiveSource = useSignStore(s => s.setActiveSource)

  // Tab con đang mở (upload/record) chính là nguồn KQ hiển thị ở ResultPanel.
  useEffect(() => {
    setActiveSource(activeTab)
  }, [activeTab, setActiveSource])

  // -- Upload State --
  const [uploadedVideos, setUploadedVideos] = useState([])
  // -- Record State (danh sách RIÊNG, không dùng chung với upload) --
  const [recordedVideos, setRecordedVideos] = useState([])
  const [isComposing, setIsComposing] = useState(false)
  const [isBatchProcessing, setIsBatchProcessing] = useState(false)
  const fileInputRef = useRef(null)

  const isAllChecked = uploadedVideos.length > 0 && uploadedVideos.every(v => v.checked)
  // Số video sẽ được xử lý khi bấm "Xử lý đã chọn" (đã chọn VÀ còn chờ xử lý).
  const pendingSelectedCount = uploadedVideos.filter(v => v.checked && v.status === 'pending').length

  // -- Recording State --
  const [recordingState, setRecordingState] = useState('idle') // 'idle' | 'recording' | 'preview'
  // Stream giữ trong ref (không phải state): các hàm cleanup/effect luôn thấy stream
  // hiện tại, không bị "đóng băng" giá trị cũ (closure) -> camera được tắt chắc chắn.
  const recordStreamRef = useRef(null)
  // Token phiên camera: stopCamera() tăng token để huỷ các lần mở camera đang chờ
  // (getUserMedia resolve SAU khi đã rời tab sẽ bị dừng ngay, không giữ camera bật).
  const camSessionRef = useRef(0)
  const [recordedBlob, setRecordedBlob] = useState(null)
  const [recordedUrl, setRecordedUrl] = useState('')
  const [recordTime, setRecordTime] = useState(0)
  const recordVideoRef = useRef(null)
  const mediaRecorderRef = useRef(null)
  const recordChunksRef = useRef([])
  const timerIntervalRef = useRef(null)

  // --- Upload Handlers ---
  const handleFileChange = (e) => {
    const files = Array.from(e.target.files)
    if (!files.length) return
    addFilesToList(files)
  }

  const getVideoDuration = (file) => {
    return new Promise((resolve) => {
      const video = document.createElement('video')
      video.preload = 'metadata'
      const url = URL.createObjectURL(file)
      const finish = (duration) => {
        URL.revokeObjectURL(url)
        resolve(duration)
      }
      video.onloadedmetadata = () => finish(video.duration)
      video.onerror = () => finish(0)
      video.src = url
    })
  }

  const addFilesToList = async (files) => {
    const validFiles = files.filter(f => f.type.startsWith('video/'))
    if (!validFiles.length) return

    const newItems = await Promise.all(
      validFiles.map(async (file, index) => {
        let durationStr = '--:--'
        try {
          const dur = await getVideoDuration(file)
          if (dur > 0) {
            const mins = Math.floor(dur / 60)
            const secs = Math.floor(dur % 60)
            durationStr = `${mins}:${secs.toString().padStart(2, '0')}`
          }
        } catch (e) {
          console.warn('Lỗi đọc độ dài video:', e)
        }

        return {
          id: `upload-${Date.now()}-${index}`,
          file,
          name: file.name,
          size: (file.size / (1024 * 1024)).toFixed(2) + ' MB',
          duration: durationStr,
          url: URL.createObjectURL(file),
          status: 'pending', // 'pending' | 'processing' | 'completed' | 'error'
          currentStep: '',
          checked: true, // Default to true (checked)
          words: [], // các từ nhận dạng được từ video này
          errorMsg: ''
        }
      })
    )

    setUploadedVideos(prev => [...prev, ...newItems])
  }

  const toggleCheckVideo = (id) => {
    setUploadedVideos(prev => prev.map(v => v.id === id ? { ...v, checked: !v.checked } : v))
  }

  const toggleCheckAll = () => {
    const isAllChecked = uploadedVideos.length > 0 && uploadedVideos.every(v => v.checked)
    const nextVal = !isAllChecked
    setUploadedVideos(prev => prev.map(v => ({ ...v, checked: nextVal })))
  }

  const processSelectedVideos = async () => {
    if (isBatchProcessing) return
    // Chốt danh sách cần xử lý ngay lúc bấm; xử lý TẤT CẢ trong một lần bấm.
    const selected = uploadedVideos.filter(v => v.checked && v.status === 'pending')
    if (!selected.length) return
    setIsBatchProcessing(true)
    try {
      // Tuần tự để tránh dồn nhiều video nặng lên backend cùng lúc.
      for (const v of selected) {
        await recognizeInto(setUploadedVideos, v)
      }
    } finally {
      setIsBatchProcessing(false)
    }
  }

  const handleDragOver = (e) => {
    e.preventDefault()
  }

  const handleDrop = (e) => {
    e.preventDefault()
    const files = Array.from(e.dataTransfer.files)
    addFilesToList(files)
  }

  const removeUploadedVideo = (id) => {
    setUploadedVideos(prev => {
      const item = prev.find(v => v.id === id)
      if (item && item.url) URL.revokeObjectURL(item.url)
      return prev.filter(v => v.id !== id)
    })
  }

  // Cập nhật một item trong danh sách bất kỳ (upload hoặc record).
  const updateItem = (setList, id, updates) => {
    setList(prev => prev.map(v => v.id === id ? { ...v, ...updates } : v))
  }

  // Nhận dạng một video và ghi kết quả vào đúng danh sách của nó (setList).
  const recognizeInto = async (setList, video) => {
    if (!video || video.status === 'processing') return
    const id = video.id
    updateItem(setList, id, { status: 'processing', currentStep: 'Đang nhận dạng ký hiệu...', errorMsg: '' })
    try {
      const { glosses } = await api.recognizeVideo(video.file)
      updateItem(setList, id, {
        status: 'completed',
        currentStep: 'Hoàn thành',
        words: Array.isArray(glosses) ? glosses : []
      })
    } catch (err) {
      console.error('Lỗi nhận dạng video:', err)
      updateItem(setList, id, {
        status: 'error',
        currentStep: 'Lỗi',
        errorMsg: 'Không nhận dạng được video. Vui lòng thử lại.'
      })
    }
  }

  // --- Recording Handlers ---
  const startCameraForRecord = async () => {
    try {
      stopCamera()
      const session = camSessionRef.current

      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: 'user' },
        audio: false
      })

      if (session !== camSessionRef.current) {
        stream.getTracks().forEach(t => t.stop())
        return
      }

      recordStreamRef.current = stream
      if (recordVideoRef.current) {
        recordVideoRef.current.srcObject = stream
        await recordVideoRef.current.play()
      }
      setRecordingState('idle')
    } catch (err) {
      console.error('Lỗi truy cập camera:', err)
      alert('Không thể truy cập camera. Vui lòng kiểm tra quyền thiết bị.')
    }
  }

  const stopCamera = () => {
    camSessionRef.current += 1
    if (recordStreamRef.current) {
      recordStreamRef.current.getTracks().forEach(t => t.stop())
      recordStreamRef.current = null
    }
  }

  useEffect(() => {
    if (activeTab === 'record') {
      startCameraForRecord()
    } else {
      stopCamera()
      cleanupRecording()
    }

    return () => {
      stopCamera()
      // Không để interval đếm giờ chạy tiếp sau khi rời tab/unmount giữa lúc quay.
      if (timerIntervalRef.current) {
        clearInterval(timerIntervalRef.current)
        timerIntervalRef.current = null
      }
    }
  }, [activeTab])

  const startRecording = () => {
    const stream = recordStreamRef.current
    if (!stream) return

    recordChunksRef.current = []
    const options = { mimeType: 'video/webm;codecs=vp9' }
    let mediaRecorder

    try {
      mediaRecorder = new MediaRecorder(stream, options)
    } catch {
      try {
        mediaRecorder = new MediaRecorder(stream)
      } catch (err) {
        console.error('Không khởi tạo được MediaRecorder:', err)
        return
      }
    }

    mediaRecorder.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) {
        recordChunksRef.current.push(e.data)
      }
    }

    mediaRecorder.onstop = () => {
      const blob = new Blob(recordChunksRef.current, { type: 'video/webm' })
      const url = URL.createObjectURL(blob)
      setRecordedBlob(blob)
      setRecordedUrl(url)
      setRecordingState('preview')
      stopCamera()
    }

    mediaRecorderRef.current = mediaRecorder
    mediaRecorder.start(10) // Collect data chunks every 10ms
    setRecordingState('recording')
    setRecordTime(0)

    timerIntervalRef.current = setInterval(() => {
      setRecordTime(prev => prev + 1)
    }, 1000)
  }

  const stopRecording = () => {
    if (mediaRecorderRef.current && recordingState === 'recording') {
      mediaRecorderRef.current.stop()
    }
    if (timerIntervalRef.current) {
      clearInterval(timerIntervalRef.current)
      timerIntervalRef.current = null
    }
  }

  const cleanupRecording = () => {
    if (recordedUrl) URL.revokeObjectURL(recordedUrl)
    setRecordedBlob(null)
    setRecordedUrl('')
    setRecordingState('idle')
    setRecordTime(0)
    if (timerIntervalRef.current) {
      clearInterval(timerIntervalRef.current)
      timerIntervalRef.current = null
    }
  }

  const handleRecordAgain = () => {
    cleanupRecording()
    startCameraForRecord()
  }

  const submitRecordedVideo = () => {
    if (!recordedBlob) return

    // Bản quay là MỘT tính năng RIÊNG: thêm vào danh sách record (không đẩy sang upload).
    const recordedFile = new File([recordedBlob], `TuQuay_${Date.now()}.webm`, { type: 'video/webm' })
    // URL độc lập với preview để cleanupRecording() thu hồi preview không làm hỏng thumbnail.
    const itemUrl = URL.createObjectURL(recordedBlob)
    const newItem = {
      id: `record-${Date.now()}`,
      file: recordedFile,
      name: recordedFile.name,
      size: (recordedFile.size / (1024 * 1024)).toFixed(2) + ' MB',
      duration: formatTime(recordTime),
      url: itemUrl,
      status: 'pending',
      currentStep: '',
      words: [],
      errorMsg: ''
    }

    setRecordedVideos(prev => [...prev, newItem])
    // Trở lại camera trực tiếp để ghi ký hiệu tiếp theo (giữ nguyên tab record).
    cleanupRecording()
    startCameraForRecord()
    // Nhận dạng bản vừa quay -> ghi vào danh sách record.
    recognizeInto(setRecordedVideos, newItem)
  }

  const removeRecordedVideo = (id) => {
    setRecordedVideos(prev => {
      const item = prev.find(v => v.id === id)
      if (item && item.url) URL.revokeObjectURL(item.url)
      return prev.filter(v => v.id !== id)
    })
  }

  // --- Ghép câu & đọc: gộp từ của MỘT nguồn -> câu (LLM) + giọng nói (TTS) ---
  // Ghi kết quả vào đúng nguồn (upload/record) để không đè lên nguồn khác.
  const uploadWords = uploadedVideos.filter(v => v.status === 'completed').flatMap(v => v.words || [])
  const recordWords = recordedVideos.filter(v => v.status === 'completed').flatMap(v => v.words || [])

  const composeFor = async (source, words) => {
    if (!words.length || isComposing) return
    setIsComposing(true)

    // Đổ các từ vào bucket của nguồn này trước khi ghép câu.
    clearSession(source)
    words.forEach(w => addGloss(source, w))
    setProcessing(source, true)

    try {
      const { sentence, audio_b64, mime } = await api.composeSpeech(words)
      setSentence(source, sentence || words.join(' '))
      if (audio_b64) {
        setAudioUrl(source, base64ToAudioUrl(audio_b64, mime || 'audio/wav'))
      }
    } catch (err) {
      console.error('Lỗi ghép câu & đọc:', err)
      // Vẫn hiển thị câu ghép thô nếu backend lỗi.
      setSentence(source, words.join(' '))
    } finally {
      setProcessing(source, false)
      setIsComposing(false)
    }
  }

  // Helper format for recording timer
  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`
  }

  return (
    <div className="flex-1 flex flex-col h-full bg-slate-100 dark:bg-slate-950 transition-colors duration-300 min-h-0 overflow-hidden">
      
      {/* Compact sub-header selector */}
      <div className="px-6 py-3 flex items-center justify-between border-b border-slate-200 dark:border-slate-900 bg-white/20 dark:bg-slate-950/20 backdrop-blur-sm shrink-0">
        <span className="text-xs font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider">
          {activeTab === 'upload' ? 'Tải lên video hệ thống' : 'Ghi hình camera trực tiếp'}
        </span>

        {/* Tab Controls */}
        <div className="flex p-0.5 rounded-lg bg-slate-200/60 dark:bg-slate-900/60 border border-slate-300 dark:border-slate-800">
          <button
            onClick={() => setActiveTab('upload')}
            className={`flex items-center gap-1.5 px-3 py-1 rounded-md text-xs font-semibold transition-all duration-200 ${
              activeTab === 'upload'
                ? 'bg-white dark:bg-slate-800 text-indigo-600 dark:text-indigo-400 shadow-sm'
                : 'text-slate-500 dark:text-slate-400 hover:text-slate-800 dark:hover:text-slate-200'
            }`}
          >
            <UploadCloud size={13} />
            <span>Tải lên tệp</span>
          </button>
          <button
            onClick={() => setActiveTab('record')}
            className={`flex items-center gap-1.5 px-3 py-1 rounded-md text-xs font-semibold transition-all duration-200 ${
              activeTab === 'record'
                ? 'bg-white dark:bg-slate-800 text-indigo-600 dark:text-indigo-400 shadow-sm'
                : 'text-slate-500 dark:text-slate-400 hover:text-slate-800 dark:hover:text-slate-200'
            }`}
          >
            <Camera size={13} />
            <span>Tự ghi hình</span>
          </button>
        </div>
      </div>

      {/* Main Tab Content */}
      <div className="flex-1 p-6 overflow-y-auto flex flex-col min-h-0">
        {activeTab === 'upload' ? (
          /* --- UPLOAD TAB CONTENT --- */
          <div className="flex-1 flex flex-col gap-6">
            
            {/* Drag & Drop Area */}
            <div
              onDragOver={handleDragOver}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              className="border-2 border-dashed border-slate-300 dark:border-slate-800 hover:border-indigo-500/50 dark:hover:border-indigo-400/50 rounded-2xl p-8 text-center bg-white/30 dark:bg-slate-950/20 backdrop-blur-sm cursor-pointer hover:bg-white/50 dark:hover:bg-slate-900/10 transition-all duration-300 flex flex-col items-center justify-center gap-3 shrink-0"
            >
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileChange}
                accept="video/*"
                multiple
                className="hidden"
              />
              <div className="p-4 rounded-full bg-indigo-500/5 text-indigo-500 dark:text-indigo-400 border border-indigo-500/10">
                <UploadCloud size={32} className="animate-bounce" />
              </div>
              <div>
                <p className="text-sm font-semibold text-slate-700 dark:text-slate-200">
                  Kéo thả video của bạn vào đây hoặc <span className="text-indigo-500 hover:underline">Chọn tệp</span>
                </p>
                <p className="text-xs text-slate-500 dark:text-slate-400 mt-1.5">
                  Hỗ trợ MP4, WEBM, MOV (Tối đa 100MB/tệp). Cho phép tải lên nhiều video.
                </p>
              </div>
            </div>

            {/* Uploaded List Table */}
            {uploadedVideos.length > 0 && (
              <div className="flex-1 flex flex-col min-h-0 bg-white/40 dark:bg-slate-950/25 border border-slate-200 dark:border-slate-900 rounded-2xl p-4">
                <div className="flex items-center justify-between pb-3 border-b border-slate-200 dark:border-slate-900 mb-4 shrink-0">
                  <span className="text-xs font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider">
                    Danh sách video ({uploadedVideos.length})
                  </span>
                  
                  <div className="flex gap-2">
                    {uploadedVideos.some(v => v.status === 'pending' && v.checked) && (
                      <button
                        onClick={processSelectedVideos}
                        disabled={isBatchProcessing}
                        className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-indigo-500 hover:bg-indigo-600 disabled:opacity-60 disabled:cursor-not-allowed text-xs font-semibold text-white transition-all shadow-md shadow-indigo-500/10"
                      >
                        {isBatchProcessing ? <Loader2 size={12} className="animate-spin" /> : <Sparkles size={12} />}
                        <span>{isBatchProcessing ? 'Đang xử lý...' : `Xử lý đã chọn (${pendingSelectedCount})`}</span>
                      </button>
                    )}
                    {uploadWords.length > 0 && (
                      <button
                        onClick={() => composeFor('upload', uploadWords)}
                        disabled={isComposing}
                        className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-emerald-500 hover:bg-emerald-600 disabled:opacity-60 text-xs font-semibold text-white transition-all shadow-md shadow-emerald-500/10"
                      >
                        {isComposing ? <Loader2 size={12} className="animate-spin" /> : <Play size={12} />}
                        <span>Ghép câu & đọc ({uploadWords.length})</span>
                      </button>
                    )}
                    <button
                      onClick={() => setUploadedVideos([])}
                      className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-slate-100 hover:bg-rose-500/10 dark:bg-slate-900 text-slate-500 hover:text-rose-500 border border-slate-200 dark:border-slate-800 hover:border-rose-500/20 text-xs font-semibold transition-all"
                    >
                      <Trash2 size={12} />
                      <span>Xóa hết</span>
                    </button>
                  </div>
                </div>

                {/* Table View */}
                <div className="flex-1 overflow-x-auto pr-1 min-h-0 border border-slate-200/80 dark:border-slate-800/80 rounded-xl bg-white/50 dark:bg-slate-900/10">
                  <table className="w-full text-left border-collapse text-xs">
                    <thead>
                      <tr className="border-b border-slate-200 dark:border-slate-800 bg-slate-50/50 dark:bg-slate-900/60 text-slate-500 dark:text-slate-400 font-bold uppercase tracking-wider select-none">
                        <th className="py-3 px-4 w-10 text-center">
                          <input
                            type="checkbox"
                            checked={isAllChecked}
                            onChange={toggleCheckAll}
                            className="rounded border-slate-300 text-indigo-600 focus:ring-indigo-500 w-4 h-4 cursor-pointer"
                          />
                        </th>
                        <th className="py-3 px-2 w-12 text-center">STT</th>
                        <th className="py-3 px-4">Tiêu đề</th>
                        <th className="py-3 px-4 w-24">Kích thước</th>
                        <th className="py-3 px-4 w-24">Độ dài</th>
                        <th className="py-3 px-4 w-40">Trạng thái</th>
                        <th className="py-3 px-4">Từ nhận dạng</th>
                        <th className="py-3 px-4 w-12 text-center">Xóa</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
                      {uploadedVideos.map((v, index) => (
                        <tr 
                          key={v.id} 
                          className={`hover:bg-slate-50/50 dark:hover:bg-slate-900/10 transition-colors ${
                            v.checked ? 'bg-indigo-50/5 dark:bg-indigo-950/10' : ''
                          }`}
                        >
                          <td className="py-3 px-4 text-center">
                            <input
                              type="checkbox"
                              checked={v.checked}
                              onChange={() => toggleCheckVideo(v.id)}
                              disabled={v.status === 'processing'}
                              className="rounded border-slate-300 text-indigo-600 focus:ring-indigo-500 w-4 h-4 cursor-pointer disabled:opacity-50"
                            />
                          </td>
                          <td className="py-3 px-2 text-center text-slate-500 dark:text-slate-400 font-medium">
                            {index + 1}
                          </td>
                          <td className="py-2 px-4 min-w-[220px]">
                            <div className="flex items-center gap-3">
                              {/* Video thumbnail preview */}
                              <div className="relative w-12 h-9 rounded bg-slate-900 overflow-hidden border border-slate-200 dark:border-slate-800 shrink-0 flex items-center justify-center">
                                <video src={v.url} className="w-full h-full object-cover" muted />
                                <div className="absolute inset-0 bg-slate-950/30 flex items-center justify-center opacity-0 hover:opacity-100 transition-opacity">
                                  <Play size={10} className="text-white fill-white" />
                                </div>
                              </div>
                              {/* Video title text */}
                              <span className="font-semibold text-slate-700 dark:text-slate-200 truncate max-w-[180px]" title={v.name}>
                                {v.name}
                              </span>
                            </div>
                          </td>
                          <td className="py-3 px-4 text-slate-500 dark:text-slate-400 font-medium">
                            {v.size}
                          </td>
                          <td className="py-3 px-4 text-slate-500 dark:text-slate-400 font-medium">
                            {v.duration}
                          </td>
                          <td className="py-3 px-4">
                            {v.status === 'pending' && (
                              <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400 font-bold">
                                Chờ xử lý
                              </span>
                            )}
                            {v.status === 'processing' && (
                              <span className="inline-flex items-center gap-1.5 text-indigo-500 dark:text-indigo-400 font-bold animate-pulse text-[10px]">
                                <Loader2 size={10} className="animate-spin" />
                                <span className="truncate">{v.currentStep}</span>
                              </span>
                            )}
                            {v.status === 'completed' && (
                              <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-md bg-emerald-500/10 text-emerald-500 dark:text-emerald-400 border border-emerald-500/20 font-bold">
                                <CheckCircle size={10} />
                                <span>Hoàn thành</span>
                              </span>
                            )}
                            {v.status === 'error' && (
                              <span
                                className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-md bg-rose-500/10 text-rose-500 dark:text-rose-400 border border-rose-500/20 font-bold"
                                title={v.errorMsg}
                              >
                                <AlertCircle size={10} />
                                <span>Lỗi</span>
                              </span>
                            )}
                          </td>
                          <td className="py-3 px-4 min-w-[160px] text-slate-700 dark:text-slate-200">
                            {v.status === 'completed' ? (
                              v.words && v.words.length > 0 ? (
                                <span className="font-medium">{v.words.join('  •  ')}</span>
                              ) : (
                                <span className="text-slate-400 dark:text-slate-500 italic">Không nhận dạng được từ nào</span>
                              )
                            ) : (
                              <span className="text-slate-400 dark:text-slate-600">—</span>
                            )}
                          </td>
                          <td className="py-3 px-4 text-center">
                            <button
                              onClick={() => removeUploadedVideo(v.id)}
                              disabled={v.status === 'processing'}
                              className="p-1.5 rounded-lg bg-slate-100 hover:bg-rose-500/10 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 text-slate-500 hover:text-rose-500 hover:border-rose-500/20 transition-all disabled:opacity-50"
                              title="Xóa video"
                            >
                              <Trash2 size={12} />
                            </button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {/* Empty State */}
            {uploadedVideos.length === 0 && (
              <div className="flex-1 flex flex-col items-center justify-center text-center p-8 bg-white/20 dark:bg-slate-950/10 border border-slate-200 dark:border-slate-900 rounded-2xl">
                <FileVideo size={48} className="text-slate-400 dark:text-slate-600 mb-3" />
                <h3 className="text-sm font-semibold text-slate-600 dark:text-slate-300">
                  Chưa tải lên video nào
                </h3>
                <p className="text-xs text-slate-500 dark:text-slate-400 mt-1 max-w-xs">
                  Tải lên một hoặc nhiều tệp video ký hiệu để hệ thống xử lý dịch sang văn bản & giọng nói.
                </p>
              </div>
            )}

          </div>
        ) : (
          /* --- RECORD TAB CONTENT: camera full-bleed + dải kết quả bản ghi (RIÊNG) --- */
          <div className="flex-1 flex flex-col min-h-0 gap-3">
          <div className="relative flex-1 bg-slate-100 dark:bg-slate-950 flex items-center justify-center overflow-hidden transition-colors duration-300 group min-h-0 rounded-2xl">
            {recordingState !== 'preview' ? (
              /* Live Camera view (mirror effect). key riêng để React KHÔNG tái dùng
                 chung DOM node với video preview — nếu dùng chung, srcObject (stream)
                 vẫn còn và được ưu tiên hơn src, khiến bản ghi không xem lại được. */
              <video
                key="live"
                ref={recordVideoRef}
                className="w-full h-full object-cover scale-x-[-1]"
                muted
                playsInline
              />
            ) : (
              /* Recorded video preview player */
              <video
                key="preview"
                src={recordedUrl}
                className="w-full h-full object-cover"
                controls
                playsInline
              />
            )}

            {/* Status Overlay for Recording */}
            {recordingState === 'recording' && (
              <div className="absolute top-4 left-4 z-20 flex items-center gap-2 bg-rose-500/90 text-white backdrop-blur-md px-3 py-1.5 rounded-full border border-rose-400 text-[10px] font-bold uppercase tracking-wider shadow-md animate-pulse">
                <span className="relative flex h-2 w-2">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-white opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-white"></span>
                </span>
                <span>RECORDING ({formatTime(recordTime)})</span>
              </div>
            )}

            {/* Status Overlay for Preview */}
            {recordingState === 'preview' && (
              <div className="absolute top-4 left-4 z-20 flex items-center gap-1.5 bg-emerald-500/90 text-white backdrop-blur-md px-3 py-1.5 rounded-full border border-emerald-400 text-[10px] font-bold uppercase tracking-wider shadow-md">
                <CheckCircle size={12} />
                <span>BẢN GHI ĐÃ SẴN SÀNG</span>
              </div>
            )}

            {/* Overlaid Recorder Controls centered at bottom */}
            <div className="absolute bottom-6 left-1/2 transform -translate-x-1/2 z-20 flex gap-3">
              {recordingState === 'idle' && (
                <button
                  onClick={startRecording}
                  className="flex items-center gap-2 px-6 py-3 rounded-full bg-rose-600 hover:bg-rose-700 text-xs font-bold text-white shadow-xl shadow-rose-600/30 transition-all scale-105 active:scale-95"
                >
                  <span className="w-2 h-2 rounded-full bg-white animate-ping mr-0.5" />
                  <span>Bắt đầu quay</span>
                </button>
              )}

              {recordingState === 'recording' && (
                <button
                  onClick={stopRecording}
                  className="flex items-center gap-2 px-6 py-3 rounded-full bg-slate-900/90 hover:bg-slate-950 border border-slate-700 text-xs font-bold text-rose-500 shadow-xl transition-all active:scale-95 animate-[pulse_1.5s_infinite]"
                >
                  <Square size={13} className="fill-rose-500" />
                  <span>Dừng quay ({formatTime(recordTime)})</span>
                </button>
              )}

              {recordingState === 'preview' && (
                <div className="flex gap-3 bg-slate-900/50 backdrop-blur-md p-1.5 rounded-full border border-slate-800">
                  <button
                    onClick={handleRecordAgain}
                    className="flex items-center gap-2 px-5 py-2 rounded-full bg-slate-900/95 hover:bg-slate-950 border border-slate-800 text-slate-200 text-xs font-semibold transition-all"
                  >
                    <RefreshCw size={13} />
                    <span>Quay lại</span>
                  </button>
                  <button
                    onClick={submitRecordedVideo}
                    className="flex items-center gap-2 px-6 py-2 rounded-full bg-indigo-500 hover:bg-indigo-600 text-xs font-bold text-white shadow-xl shadow-indigo-500/20 transition-all"
                  >
                    <Send size={13} />
                    <span>Nhận dạng ký hiệu này</span>
                  </button>
                </div>
              )}
            </div>
            </div>{/* hết vùng camera */}

            {/* Dải các ký hiệu đã ghi (kết quả RIÊNG của tab record) */}
            {recordedVideos.length > 0 && (
              <div className="shrink-0 bg-white/40 dark:bg-slate-950/25 border border-slate-200 dark:border-slate-900 rounded-2xl p-3">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider">
                    Ký hiệu đã ghi ({recordedVideos.length})
                  </span>
                  <div className="flex gap-2">
                    {recordWords.length > 0 && (
                      <button
                        onClick={() => composeFor('record', recordWords)}
                        disabled={isComposing}
                        className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-emerald-500 hover:bg-emerald-600 disabled:opacity-60 text-xs font-semibold text-white transition-all"
                      >
                        {isComposing ? <Loader2 size={12} className="animate-spin" /> : <Play size={12} />}
                        <span>Ghép câu & đọc ({recordWords.length})</span>
                      </button>
                    )}
                    <button
                      onClick={() => {
                        recordedVideos.forEach(v => v.url && URL.revokeObjectURL(v.url))
                        setRecordedVideos([])
                      }}
                      className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-slate-100 hover:bg-rose-500/10 dark:bg-slate-900 text-slate-500 hover:text-rose-500 border border-slate-200 dark:border-slate-800 text-xs font-semibold transition-all"
                    >
                      <Trash2 size={12} />
                      <span>Xóa hết</span>
                    </button>
                  </div>
                </div>
                <div className="flex gap-2 overflow-x-auto pb-1">
                  {recordedVideos.map((v) => (
                    <div key={v.id} className="shrink-0 w-40 rounded-xl border border-slate-200 dark:border-slate-800 bg-white/50 dark:bg-slate-900/20 p-2 flex flex-col gap-1.5">
                      <div className="relative">
                        <video src={v.url} className="w-full h-20 object-cover rounded-lg bg-slate-900" muted />
                        <button
                          onClick={() => removeRecordedVideo(v.id)}
                          className="absolute top-1 right-1 p-1 rounded-md bg-slate-950/60 text-white hover:bg-rose-500/80 transition-colors"
                          title="Xóa"
                        >
                          <Trash2 size={11} />
                        </button>
                      </div>
                      <div className="text-[11px] font-semibold min-h-[16px] leading-tight">
                        {v.status === 'processing' && (
                          <span className="text-indigo-500 dark:text-indigo-400 inline-flex items-center gap-1">
                            <Loader2 size={10} className="animate-spin" />Đang nhận dạng...
                          </span>
                        )}
                        {v.status === 'completed' && (
                          v.words && v.words.length > 0
                            ? <span className="text-slate-700 dark:text-slate-200">{v.words.join('  •  ')}</span>
                            : <span className="text-slate-400 dark:text-slate-500 italic">Không nhận dạng được từ nào</span>
                        )}
                        {v.status === 'error' && (
                          <span className="text-rose-500 dark:text-rose-400 inline-flex items-center gap-1">
                            <AlertCircle size={10} />Lỗi
                          </span>
                        )}
                        {v.status === 'pending' && <span className="text-slate-400 dark:text-slate-500">Chờ xử lý...</span>}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

      </div>
    </div>
  )
}
