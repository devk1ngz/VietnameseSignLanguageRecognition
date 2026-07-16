import { Wifi, WifiOff, RefreshCw, AlertCircle, Video, UploadCloud, Sun, Moon } from 'lucide-react'
import logoDark from '../../assets/logo_dark.png'
import logoLight from '../../assets/logo_light.png'
import useSignStore from '../../store/useSignStore'

export default function StatusBar({ status, activeTab, setActiveTab, onBack }) {
  const theme = useSignStore(s => s.theme)
  const toggleTheme = useSignStore(s => s.toggleTheme)
  const logoSrc = theme === 'dark' ? logoLight : logoDark

  const statusConfig = {
    connected: {
      text: 'Đang kết nối (Thời gian thực)',
      bg: 'bg-emerald-500/10 border-emerald-500/20 text-emerald-600 dark:text-emerald-400',
      icon: <Wifi size={15} className="animate-pulse" />,
      dotClass: 'bg-emerald-400 shadow-[0_0_8px_#34d399]'
    },
    connecting: {
      text: 'Đang kết nối máy chủ...',
      bg: 'bg-amber-500/10 border-amber-500/20 text-amber-600 dark:text-amber-400',
      icon: <RefreshCw size={15} className="animate-spin" />,
      dotClass: 'bg-amber-400 shadow-[0_0_8px_#fbbf24]'
    },
    disconnected: {
      text: 'Mất kết nối',
      bg: 'bg-slate-500/10 border-slate-500/20 text-slate-600 dark:text-slate-400',
      icon: <WifiOff size={15} />,
      dotClass: 'bg-slate-400'
    },
    error: {
      text: 'Lỗi kết nối máy chủ',
      bg: 'bg-rose-500/10 border-rose-500/20 text-rose-600 dark:text-rose-400',
      icon: <AlertCircle size={15} />,
      dotClass: 'bg-rose-400 shadow-[0_0_8px_#f87171]'
    }
  }

  const current = statusConfig[status] || statusConfig.disconnected

  return (
    <header className="w-full border-b border-slate-200/60 dark:border-slate-900/60 bg-white/60 dark:bg-slate-950/60 backdrop-blur-md select-none shrink-0 transition-colors duration-300 z-50">
      <div className="w-full px-8 lg:px-[5%] py-4 md:py-0 md:h-20 flex flex-col md:flex-row md:items-center justify-between gap-4">
        
        {/* 1. Left Section: Clickable Logo & Brand Description */}
        <div 
          onClick={onBack}
          className="flex items-center gap-3 cursor-pointer group select-none min-w-[260px]"
          title="Quay lại trang chủ"
        >
          <img src={logoSrc} alt="Logo" className="h-10 md:h-12 w-auto object-contain rounded-xl transition-all duration-300 group-hover:scale-105" />
          <div className="flex flex-col">
            <span className="text-lg font-bold tracking-tight text-slate-800 dark:text-white bg-gradient-to-r from-indigo-500 to-purple-500 bg-clip-text text-transparent group-hover:opacity-80 transition-all">
              SignSpeak AI
            </span>
            <span className="text-[10px] text-slate-500 dark:text-slate-400 font-medium tracking-wide mt-0.5">
              Hệ thống nhận dạng Ký hiệu Tiếng Việt
            </span>
          </div>
        </div>

        {/* 2. Center Section: Tab Switcher (Symmetrical, Simple design, Larger font) */}
        <div className="flex justify-center items-center flex-1">
          <div className="flex p-1 rounded-2xl bg-slate-200/50 dark:bg-slate-900/50 border border-slate-300 dark:border-slate-800 shadow-inner w-full max-w-md">
            <button
              onClick={() => setActiveTab('realtime')}
              className={`flex items-center justify-center gap-2 flex-1 py-2.5 rounded-xl text-sm font-bold transition-all duration-200 ${
                activeTab === 'realtime'
                  ? 'bg-white dark:bg-slate-800 text-indigo-600 dark:text-indigo-400 shadow-sm border border-slate-200/20 dark:border-slate-700/50'
                  : 'text-slate-500 hover:text-slate-800 dark:text-slate-400 dark:hover:text-slate-200'
              }`}
            >
              <Video size={16} />
              <span>Nhận dạng realtime</span>
            </button>
            
            <button
              onClick={() => setActiveTab('manual')}
              className={`flex items-center justify-center gap-2 flex-1 py-2.5 rounded-xl text-sm font-bold transition-all duration-200 ${
                activeTab === 'manual'
                  ? 'bg-white dark:bg-slate-800 text-indigo-600 dark:text-indigo-400 shadow-sm border border-slate-200/20 dark:border-slate-700/50'
                  : 'text-slate-500 hover:text-slate-800 dark:text-slate-400 dark:hover:text-slate-200'
              }`}
            >
              <UploadCloud size={16} />
              <span>Nhận dạng thủ công</span>
            </button>
          </div>
        </div>

        {/* 3. Right Section: Connection Status & Theme Toggle */}
        <div className="flex justify-start md:justify-end items-center gap-3 min-w-[280px]">
          <div className={`flex items-center gap-2 px-4 py-2 rounded-full border text-xs md:text-sm font-semibold transition-all duration-300 ${current.bg}`}>
            <span className={`w-2 h-2 rounded-full ${current.dotClass} mr-0.5`} />
            {current.icon}
            <span>{current.text}</span>
          </div>

          {/* Theme Toggle Button */}
          <button
            onClick={toggleTheme}
            className="p-2.5 rounded-xl text-slate-500 hover:text-indigo-500 dark:text-slate-400 dark:hover:text-indigo-400 bg-slate-100 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 transition-all duration-200 shadow-sm"
            title={theme === 'dark' ? 'Chuyển sang giao diện sáng' : 'Chuyển sang giao diện tối'}
          >
            {theme === 'dark' ? <Sun size={16} /> : <Moon size={16} />}
          </button>
        </div>

      </div>
    </header>
  )
}
