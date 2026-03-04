import {
  useEffect,
  useRef,
  useState,
  type MouseEvent as ReactMouseEvent,
  type SyntheticEvent,
  type WheelEvent as ReactWheelEvent,
} from 'react'
import { Alert, Box, CircularProgress, Stack, Typography } from '@mui/material'

import {
  apiClient,
  type Axis,
  type SliceQuery,
  getApiErrorMessage,
} from '../../services/api'

interface SliceViewProps {
  accent: string
  axis: Axis
  crosshair: { x: number; y: number }
  disabled?: boolean
  errorText?: string | null
  index: number
  maxIndex: number
  query: SliceQuery
  requestKey: string | null
  wl: number
  onCrosshairChange: (point: { x: number; y: number }) => void
  onSliceChange: (index: number) => void
  onWindowLevelDrag: (
    startWw: number,
    startWl: number,
    deltaX: number,
    deltaY: number,
  ) => void
  ww: number
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/**
 * Computes the rendered image content rect (as fractions of the container)
 * for an `object-fit: contain` image. The image is centred with letterbox
 * bars on the short axis.
 */
function computeContentRect(
  containerW: number,
  containerH: number,
  naturalW: number,
  naturalH: number,
): { left: number; top: number; width: number; height: number } | null {
  if (containerW === 0 || containerH === 0 || naturalW === 0 || naturalH === 0) {
    return null
  }
  const scale = Math.min(containerW / naturalW, containerH / naturalH)
  const rw = naturalW * scale
  const rh = naturalH * scale
  return {
    left: (containerW - rw) / 2 / containerW,
    top: (containerH - rh) / 2 / containerH,
    width: rw / containerW,
    height: rh / containerH,
  }
}

function SliceView({
  accent,
  axis,
  crosshair,
  disabled = false,
  errorText,
  index,
  maxIndex,
  onCrosshairChange,
  onSliceChange,
  onWindowLevelDrag,
  query,
  requestKey,
  wl,
  ww,
}: SliceViewProps) {
  // ── Stale-while-revalidate state ─────────────────────────────────────────
  const [requestState, setRequestState] = useState<{
    key: string | null
    url: string | null
    error: string | null
  }>({ key: null, url: null, error: null })
  const [naturalSize, setNaturalSize] = useState<{ width: number; height: number }>({
    width: 0,
    height: 0,
  })
  // Rendered image content rect as fractions of the viewport container.
  // This resolves letterbox offsets so crosshairs and clicks are accurate.
  const [contentRect, setContentRect] = useState<{
    left: number
    top: number
    width: number
    height: number
  } | null>(null)

  const viewportRef = useRef<HTMLDivElement | null>(null)
  const dragStateRef = useRef<{
    startX: number
    startY: number
    startWw: number
    startWl: number
  } | null>(null)
  const prevUrlRef = useRef<string | null>(null)
  // Keep a stable ref to naturalSize so the ResizeObserver can read it without
  // re-subscribing every time naturalSize changes.
  const naturalSizeRef = useRef(naturalSize)

  useEffect(() => {
    naturalSizeRef.current = naturalSize
  }, [naturalSize])

  const queryKey = JSON.stringify(query)
  const fetchKey = !disabled && requestKey ? `${requestKey}:${axis}:${index}:${queryKey}` : null

  const displayUrl = requestState.url
  const isStale = Boolean(fetchKey) && requestState.key !== fetchKey
  const isFirstLoad = !displayUrl && Boolean(fetchKey) && requestState.key !== fetchKey
  const loadError = requestState.error && requestState.key === fetchKey ? requestState.error : null

  // ── Slice fetch ───────────────────────────────────────────────────────────
  useEffect(() => {
    if (!fetchKey) return
    if (requestState.key === fetchKey) return

    let active = true
    apiClient.getSliceBlob(axis, index, query)
      .then((blob) => {
        if (!active) return
        if (prevUrlRef.current) URL.revokeObjectURL(prevUrlRef.current)
        const nextUrl = URL.createObjectURL(blob)
        prevUrlRef.current = nextUrl
        setRequestState({ key: fetchKey, url: nextUrl, error: null })
      })
      .catch((requestError) => {
        if (!active) return
        setRequestState((prev) => ({
          ...prev,
          key: fetchKey,
          error: getApiErrorMessage(requestError),
        }))
      })
    return () => { active = false }
  }, [axis, fetchKey, index, query, requestState.key])

  useEffect(() => {
    return () => {
      if (prevUrlRef.current) { URL.revokeObjectURL(prevUrlRef.current); prevUrlRef.current = null }
    }
  }, [])

  // ── ResizeObserver: keep contentRect in sync with panel size ─────────────
  useEffect(() => {
    const el = viewportRef.current
    if (!el) return

    const observer = new ResizeObserver((entries) => {
      const entry = entries[0]
      if (!entry) return
      const { width, height } = entry.contentRect
      const { width: nw, height: nh } = naturalSizeRef.current
      setContentRect(computeContentRect(width, height, nw, nh))
    })
    observer.observe(el)
    return () => observer.disconnect()
  }, []) // intentionally empty – viewportRef.current is stable

  // Recompute contentRect whenever naturalSize changes (new image loaded)
  useEffect(() => {
    const el = viewportRef.current
    if (!el || naturalSize.width === 0) return
    const { width, height } = el.getBoundingClientRect()
    setContentRect(computeContentRect(width, height, naturalSize.width, naturalSize.height))
  }, [naturalSize])

  // ── Window/Level drag ─────────────────────────────────────────────────────
  useEffect(() => {
    function handleMouseMove(event: MouseEvent) {
      const dragState = dragStateRef.current
      if (!dragState) return
      onWindowLevelDrag(dragState.startWw, dragState.startWl,
        event.clientX - dragState.startX, event.clientY - dragState.startY)
    }
    function handleMouseUp() { dragStateRef.current = null }
    window.addEventListener('mousemove', handleMouseMove)
    window.addEventListener('mouseup', handleMouseUp)
    return () => {
      window.removeEventListener('mousemove', handleMouseMove)
      window.removeEventListener('mouseup', handleMouseUp)
    }
  }, [onWindowLevelDrag])

  function handleWheel(event: ReactWheelEvent<HTMLDivElement>) {
    event.preventDefault()
    if (disabled || maxIndex <= 0) return
    const delta = event.deltaY > 0 ? 1 : -1
    const nextIndex = Math.min(maxIndex, Math.max(0, index + delta))
    if (nextIndex !== index) onSliceChange(nextIndex)
  }

  function handleMouseDown(event: ReactMouseEvent<HTMLDivElement>) {
    if (event.button !== 2) return
    event.preventDefault()
    dragStateRef.current = { startX: event.clientX, startY: event.clientY, startWw: ww, startWl: wl }
  }

  // ── Click → crosshair (accounts for letterbox offsets) ───────────────────
  function handleViewportClick(event: ReactMouseEvent<HTMLDivElement>) {
    const rect = viewportRef.current?.getBoundingClientRect()
    if (!rect || naturalSize.width === 0) return
    const scale = Math.min(rect.width / naturalSize.width, rect.height / naturalSize.height)
    const rw = naturalSize.width * scale
    const rh = naturalSize.height * scale
    const imgLeft = rect.left + (rect.width - rw) / 2
    const imgTop = rect.top + (rect.height - rh) / 2
    const x = (event.clientX - imgLeft) / rw
    const y = (event.clientY - imgTop) / rh
    onCrosshairChange({
      x: Math.min(1, Math.max(0, x)),
      y: Math.min(1, Math.max(0, y)),
    })
  }

  // ── Crosshair overlay geometry ───────────────────────────────────────────
  // Crosshair lines are positioned within the image content area (no letterbox)
  const cr = contentRect ?? { left: 0, top: 0, width: 1, height: 1 }
  // Convert crosshair fraction into position relative to the full viewport
  const chLeft = `${(cr.left + crosshair.x * cr.width) * 100}%`
  const chTop = `${(cr.top + crosshair.y * cr.height) * 100}%`
  // Image content bounds for the crosshair lines' extents
  const crLeft = `${cr.left * 100}%`
  const crTop = `${cr.top * 100}%`
  const crRight = `${(1 - cr.left - cr.width) * 100}%`
  const crBottom = `${(1 - cr.top - cr.height) * 100}%`

  const displayNaturalSize =
    naturalSize.width > 0 ? `${naturalSize.width}×${naturalSize.height}` : ''

  return (
    <Box
      data-slice-root={axis}
      onContextMenu={(event) => event.preventDefault()}
      onMouseDown={handleMouseDown}
      onWheel={handleWheel}
      sx={{
        position: 'relative',
        flex: 1,
        minHeight: 0,
        borderRadius: 1,
        backgroundColor: '#000',
        border: '1px solid',
        borderColor: 'divider',
        overflow: 'hidden',
      }}
    >
      {errorText ? (
        <Alert severity="warning" sx={{ m: 1 }}>{errorText}</Alert>
      ) : null}

      {!errorText ? (
        <Box
          ref={viewportRef}
          data-slice-viewport={axis}
          onClick={displayUrl ? handleViewportClick : undefined}
          sx={{
            position: 'absolute',
            inset: 0,
            backgroundColor: '#000',
            cursor: displayUrl ? 'crosshair' : 'default',
          }}
        >
          {displayUrl ? (
            <>
              {/* ── Image: fills panel, aspect-ratio preserved via object-fit ── */}
              <Box
                component="img"
                src={displayUrl}
                alt={`${axis} slice ${index + 1}`}
                data-slice-axis={axis}
                data-slice-index={index}
                onLoad={(event: SyntheticEvent<HTMLImageElement>) => {
                  const { naturalWidth: nw, naturalHeight: nh } = event.currentTarget
                  setNaturalSize({ width: nw, height: nh })
                }}
                sx={{
                  display: 'block',
                  width: '100%',
                  height: '100%',
                  objectFit: 'contain',
                  userSelect: 'none',
                  opacity: isStale ? 0.65 : 1,
                  transition: 'opacity 0.12s ease-out',
                  imageRendering: 'pixelated',
                }}
              />

              {/* ── Crosshair lines, clipped to the image content area ─────── */}
              {contentRect ? (
                <Box sx={{ position: 'absolute', inset: 0, pointerEvents: 'none' }}>
                  {/* Vertical line */}
                  <Box sx={{
                    position: 'absolute',
                    left: chLeft,
                    top: crTop,
                    bottom: crBottom,
                    width: '1.5px',
                    transform: 'translateX(-50%)',
                    backgroundColor: accent,
                    opacity: 0.85,
                  }} />
                  {/* Horizontal line */}
                  <Box sx={{
                    position: 'absolute',
                    left: crLeft,
                    right: crRight,
                    top: chTop,
                    height: '1.5px',
                    transform: 'translateY(-50%)',
                    backgroundColor: accent,
                    opacity: 0.85,
                  }} />
                </Box>
              ) : null}

              {/* Stale indicator */}
              {isStale ? (
                <CircularProgress size={20} thickness={5} sx={{
                  position: 'absolute', top: 8, right: 8,
                  color: accent, opacity: 0.7,
                }} />
              ) : null}
            </>
          ) : isFirstLoad ? (
            <Stack spacing={1} alignItems="center" justifyContent="center" sx={{ height: '100%' }}>
              <CircularProgress size={24} />
              <Typography variant="body2" color="text.secondary">
                Loading {axis} slice...
              </Typography>
            </Stack>
          ) : loadError ? (
            <Box sx={{ p: 2 }}><Alert severity="error">{loadError}</Alert></Box>
          ) : (
            <Stack spacing={1} alignItems="center" justifyContent="center" sx={{ height: '100%' }}>
              <Typography variant="body2" color="text.secondary">
                Select a series to view {axis} slices.
              </Typography>
            </Stack>
          )}
        </Box>
      ) : null}

      <Stack
        direction="row"
        spacing={1}
        justifyContent="space-between"
        sx={{ position: 'absolute', left: 8, right: 8, bottom: 6, pointerEvents: 'none' }}
      >
        <Typography variant="caption" sx={{ color: 'rgba(255,255,255,0.4)', fontSize: '0.65rem' }}>
          Scroll to navigate
        </Typography>
        <Typography variant="caption" sx={{ color: 'rgba(255,255,255,0.4)', fontSize: '0.65rem' }}>
          {displayNaturalSize}
        </Typography>
      </Stack>
    </Box>
  )
}

export default SliceView
