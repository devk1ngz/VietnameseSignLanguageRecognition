
export default function GlossBadgeList({ glosses }) {
  if (!glosses.length) {
    return (
      <div className="py-3.5 text-center text-sm text-slate-400 dark:text-slate-500 italic">
        Chờ ký hiệu...
      </div>
    )
  }

  return (
    <div className="flex flex-wrap gap-2.5 max-h-[160px] overflow-y-auto p-1.5">
      {glosses.map((g, idx) => {
        // Highlight the latest gloss badge to draw attention to it
        const isLatest = idx === glosses.length - 1
        return (
          <span
            key={g.id}
            className={`px-4 py-2 text-sm font-bold rounded-xl tracking-wider transition-all duration-300 transform select-none ${
              isLatest
                ? 'bg-indigo-500 text-white border border-indigo-400 shadow-[0_0_12px_rgba(99,102,241,0.4)] scale-105 animate-[pulse_1.5s_infinite]'
                : 'bg-slate-900 hover:bg-slate-800 text-indigo-300 border border-slate-800 hover:border-slate-700'
            }`}
          >
            {g.text}
          </span>
        )
      })}
    </div>
  )
}
