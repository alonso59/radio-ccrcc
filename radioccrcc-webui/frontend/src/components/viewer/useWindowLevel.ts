import { useState } from 'react'

export const WINDOW_LEVEL_PRESETS = {
  softTissue: [400, 50],
  bone: [1800, 400],
  lung: [1500, -600],
  brain: [80, 40],
} as const

function clampWidth(value: number): number {
  return Math.max(1, Math.round(value))
}

function clampLevel(value: number): number {
  return Math.round(value)
}

export function useWindowLevel() {
  const [state, setState] = useState<{ ww: number; wl: number }>(() => {
    const [ww, wl] = WINDOW_LEVEL_PRESETS.softTissue
    return { ww, wl }
  })

  function applyPreset(preset: keyof typeof WINDOW_LEVEL_PRESETS) {
    const [ww, wl] = WINDOW_LEVEL_PRESETS[preset]
    setState({ ww, wl })
  }

  function setWindowLevel(ww: number, wl: number) {
    setState({
      ww: clampWidth(ww),
      wl: clampLevel(wl),
    })
  }

  function applyDrag(startWw: number, startWl: number, deltaX: number, deltaY: number) {
    setState({
      ww: clampWidth(startWw + deltaX * 6),
      wl: clampLevel(startWl - deltaY * 3),
    })
  }

  return {
    applyDrag,
    applyPreset,
    presets: WINDOW_LEVEL_PRESETS,
    setWindowLevel,
    wl: state.wl,
    ww: state.ww,
  }
}
