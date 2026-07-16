import { Loader2, Sparkles } from 'lucide-react'

export default function SentenceDisplay({ sentence, isProcessing }) {
  return (
    <div className={`min-h-[70px] px-5 py-4 rounded-2xl border transition-all duration-300 flex items-center ${
      isProcessing 
        ? 'bg-indigo-500/5 border-indigo-500/20' 
        : sentence
          ? 'bg-white dark:bg-slate-900 border-slate-200 dark:border-slate-800 shadow-sm'
          : 'bg-slate-100/40 dark:bg-slate-900/40 border-slate-200 dark:border-slate-800/40 border-dashed'
    }`}>
      {isProcessing ? (
        <div className="flex items-center gap-3 text-indigo-500 dark:text-indigo-400">
          <Loader2 size={18} className="animate-spin" />
          <span className="text-sm font-semibold">Đang dịch bản tin...</span>
        </div>
      ) : sentence ? (
        <div className="flex gap-3 items-center w-full">
          <Sparkles size={18} className="text-indigo-500 dark:text-indigo-400 shrink-0" />
          <p className="text-base md:text-lg text-slate-800 dark:text-slate-100 font-bold leading-relaxed">
            {sentence}
          </p>
        </div>
      ) : (
        <div className="w-full text-center py-2.5">
          <p className="text-sm text-slate-400 dark:text-slate-500 italic">Chờ câu dịch...</p>
        </div>
      )}
    </div>
  )
}
