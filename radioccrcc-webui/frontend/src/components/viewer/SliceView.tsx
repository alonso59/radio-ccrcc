import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type MouseEvent as ReactMouseEvent,
  type SyntheticEvent,
  type WheelEvent as ReactWheelEvent,
} from 'react'
import { Alert, Box, Button, CircularProgress, Stack, Typography } from '@mui/material'
import axios from 'axios'

import {
  apiClient,
  type Axis,
  type SliceQuery,
  getApiErrorMessage,
  isHandleExpiredError,
} from '../../services/api'

const MAX_SLICE_CACHE_ITEMS = 24
const MIN_ZOOM = 1
const MAX_ZOOM = 8
const ZOOM_STEP = 0.14

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
  onHandleExpired?: () => void
  onWindowLevelDrag: (
    startWw: number,
    startWl: number,
    deltaX: number,
    deltaY: number,
  ) => void
  ww: number
}

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
  onHandleExpired,
  onWindowLevelDrag,
  query,
  requestKey,
  wl,
  ww,
}: SliceViewProps) {
  const [requestState, setRequestState] = useState<{
    key: string | null
    url: string | null
    error: string | null
  }>({ key: null, url: null, error: null })
  const [naturalSize, setNaturalSize] = useState<{ width: number; height: number }>({
    width: 0,
    height: 0,
  })
  const [contentRect, setContentRect] = useState<{
    left: number
    top: number
    width: number
    height: number
  } | null>(null)
  const [viewState, setViewState] = useState<{ zoom: number; panX: number; panY: number }>({
    zoom: 1,
    panX: 0,
    panY: 0,
  })
  const [isPanning, setIsPanning] = useState(false)

  const viewportRef = useRef<HTMLDivElement | null>(null)
  const dragStateRef = useRef<{
    startX: number
    startY: number
    startWw: number
    startWl: number
  } | null>(null)
  const panDragStateRef = useRef<{
    startX: number
    startY: number
    startPanX: number
    startPanY: number
  } | null>(null)
  const suppressClickRef = useRef(false)
  const hasImageRef = useRef(false)
  const naturalSizeRef = useRef(naturalSize)
  const viewStateRef = useRef(viewState)
  const cacheRef = useRef(new Map<string, string>())
  const cacheOrderRef = useRef<string[]>([])
  const abortRef = useRef<AbortController | null>(null)
  const wheelFrameRef = useRef<number | null>(null)
  const pendingWheelDeltaRef = useRef(0)
  const indexRef = useRef(index)
  const maxIndexRef = useRef(maxIndex)

  useEffect(() => {
    naturalSizeRef.current = naturalSize
  }, [naturalSize])

  useEffect(() => {
    viewStateRef.current = viewState
  }, [viewState])

  useEffect(() => {
    indexRef.current = index
  }, [index])

  useEffect(() => {
    maxIndexRef.current = maxIndex
  }, [maxIndex])

  const queryKey = JSON.stringify(query)
  const requestQuery = useMemo(() => JSON.parse(queryKey) as SliceQuery, [queryKey])
  const fetchKey = !disabled && requestKey ? `${requestKey}:${axis}:${index}:${queryKey}` : null

  const displayUrl = requestState.url
  const isStale = Boolean(fetchKey) && requestState.key !== fetchKey
  const isFirstLoad = !displayUrl && Boolean(fetchKey) && requestState.key !== fetchKey
  const loadError = requestState.error && requestState.key === fetchKey ? requestState.error : null
  const isViewTransformed =
    Math.abs(viewState.panX) > 0.5 ||
    Math.abs(viewState.panY) > 0.5 ||
    Math.abs(viewState.zoom - 1) > 1e-3

  useEffect(() => {
    hasImageRef.current = Boolean(displayUrl)
  }, [displayUrl])

  function cacheSliceUrl(key: string, url: string) {
    const existing = cacheRef.current.get(key)
    if (existing && existing !== url) {
      URL.revokeObjectURL(existing)
    }

    cacheRef.current.set(key, url)
    cacheOrderRef.current = cacheOrderRef.current.filter((entry) => entry !== key)
    cacheOrderRef.current.push(key)

    while (cacheOrderRef.current.length > MAX_SLICE_CACHE_ITEMS) {
      const staleKey = cacheOrderRef.current.shift()
      if (!staleKey) {
        continue
      }
      const staleUrl = cacheRef.current.get(staleKey)
      if (staleUrl) {
        URL.revokeObjectURL(staleUrl)
      }
      cacheRef.current.delete(staleKey)
    }
  }

  useEffect(() => {
    if (!fetchKey) {
      return
    }
    if (requestState.key === fetchKey) {
      return
    }

    const cachedUrl = cacheRef.current.get(fetchKey)
    if (cachedUrl) {
      setRequestState({ key: fetchKey, url: cachedUrl, error: null })
      return
    }

    abortRef.current?.abort()
    const controller = new AbortController()
    abortRef.current = controller
    let active = true

    apiClient
      .getSliceBlob(axis, index, requestQuery, { signal: controller.signal })
      .then((blob) => {
        if (!active) {
          return
        }
        const nextUrl = URL.createObjectURL(blob)
        cacheSliceUrl(fetchKey, nextUrl)
        setRequestState({
          key: fetchKey,
          url: nextUrl,
          error: null,
        })
      })
      .catch((requestError) => {
        if (!active) {
          return
        }
        if (axios.isAxiosError(requestError) && requestError.code === 'ERR_CANCELED') {
          return
        }
        if (isHandleExpiredError(requestError)) {
          onHandleExpired?.()
          return
        }
        setRequestState((prev) => ({
          ...prev,
          key: fetchKey,
          error: getApiErrorMessage(requestError),
        }))
      })

    return () => {
      active = false
      controller.abort()
    }
  }, [axis, fetchKey, index, onHandleExpired, requestQuery, requestState.key])

  useEffect(() => {
    const cache = cacheRef.current
    return () => {
      abortRef.current?.abort()
      cache.forEach((url) => URL.revokeObjectURL(url))
      cache.clear()
      cacheOrderRef.current = []
      if (wheelFrameRef.current !== null) {
        window.cancelAnimationFrame(wheelFrameRef.current)
        wheelFrameRef.current = null
      }
    }
  }, [])

  useEffect(() => {
    const el = viewportRef.current
    if (!el) {
      return
    }

    const observer = new ResizeObserver((entries) => {
      const entry = entries[0]
      if (!entry) {
        return
      }
      const { width, height } = entry.contentRect
      const { width: nw, height: nh } = naturalSizeRef.current
      setContentRect(computeContentRect(width, height, nw, nh))
    })

    observer.observe(el)
    return () => observer.disconnect()
  }, [])

  useEffect(() => {
    const el = viewportRef.current
    if (!el || naturalSize.width === 0) {
      return
    }
    const { width, height } = el.getBoundingClientRect()
    setContentRect(computeContentRect(width, height, naturalSize.width, naturalSize.height))
  }, [naturalSize])

  useEffect(() => {
    function handleMouseMove(event: MouseEvent) {
      const dragState = dragStateRef.current
      if (!dragState) {
        const panDragState = panDragStateRef.current
        if (!panDragState) {
          return
        }
        const deltaX = event.clientX - panDragState.startX
        const deltaY = event.clientY - panDragState.startY
        if (Math.abs(deltaX) > 2 || Math.abs(deltaY) > 2) {
          suppressClickRef.current = true
        }
        setViewState((current) => ({
          ...current,
          panX: panDragState.startPanX + deltaX,
          panY: panDragState.startPanY + deltaY,
        }))
        return
      }
      onWindowLevelDrag(
        dragState.startWw,
        dragState.startWl,
        event.clientX - dragState.startX,
        event.clientY - dragState.startY,
      )
    }

    function handleMouseUp() {
      dragStateRef.current = null
      panDragStateRef.current = null
      setIsPanning(false)
    }

    window.addEventListener('mousemove', handleMouseMove)
    window.addEventListener('mouseup', handleMouseUp)
    return () => {
      window.removeEventListener('mousemove', handleMouseMove)
      window.removeEventListener('mouseup', handleMouseUp)
    }
  }, [onWindowLevelDrag])

  const clampZoom = useCallback((value: number): number => {
    return Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, value))
  }, [])

  const applyZoomAt = useCallback((clientX: number, clientY: number, deltaY: number) => {
    const viewport = viewportRef.current
    if (!viewport) {
      return
    }

    const rect = viewport.getBoundingClientRect()
    const prev = viewStateRef.current
    const factor = deltaY < 0 ? 1 + ZOOM_STEP : 1 / (1 + ZOOM_STEP)
    const nextZoom = clampZoom(prev.zoom * factor)
    if (Math.abs(nextZoom - prev.zoom) < 1e-6) {
      return
    }

    const centerX = rect.width / 2
    const centerY = rect.height / 2
    const cursorX = clientX - rect.left
    const cursorY = clientY - rect.top
    const baseOffsetX = (cursorX - centerX - prev.panX) / prev.zoom
    const baseOffsetY = (cursorY - centerY - prev.panY) / prev.zoom

    setViewState({
      zoom: nextZoom,
      panX: cursorX - centerX - nextZoom * baseOffsetX,
      panY: cursorY - centerY - nextZoom * baseOffsetY,
    })
  }, [clampZoom])

  function resetViewToFit() {
    setViewState({ zoom: 1, panX: 0, panY: 0 })
  }

  useEffect(() => {
    const element = viewportRef.current
    if (!element) {
      return
    }

    function handleNativeCtrlWheel(event: WheelEvent) {
      if (!(event.ctrlKey || event.metaKey) || !hasImageRef.current) {
        return
      }
      // Browser zoom on Ctrl+Wheel must be intercepted at a non-passive listener.
      event.preventDefault()
      event.stopPropagation()
      applyZoomAt(event.clientX, event.clientY, event.deltaY)
    }

    element.addEventListener('wheel', handleNativeCtrlWheel, { passive: false })
    return () => {
      element.removeEventListener('wheel', handleNativeCtrlWheel)
    }
  }, [applyZoomAt])

  function flushWheelDelta() {
    wheelFrameRef.current = null
    if (disabled || maxIndexRef.current <= 0) {
      pendingWheelDeltaRef.current = 0
      return
    }

    const currentIndex = indexRef.current
    const nextIndex = Math.min(
      maxIndexRef.current,
      Math.max(0, currentIndex + pendingWheelDeltaRef.current),
    )
    pendingWheelDeltaRef.current = 0
    if (nextIndex !== currentIndex) {
      onSliceChange(nextIndex)
    }
  }

  function handleWheel(event: ReactWheelEvent<HTMLDivElement>) {
    event.preventDefault()
    if (!displayUrl) {
      return
    }
    if (event.ctrlKey || event.metaKey) {
      // Native non-passive wheel listener handles zoom to avoid browser page zoom.
      return
    }
    if (disabled || maxIndexRef.current <= 0) {
      return
    }

    pendingWheelDeltaRef.current += event.deltaY > 0 ? 1 : -1
    if (wheelFrameRef.current === null) {
      wheelFrameRef.current = window.requestAnimationFrame(flushWheelDelta)
    }
  }

  function handleMouseDown(event: ReactMouseEvent<HTMLDivElement>) {
    if (event.button === 2) {
      event.preventDefault()
      dragStateRef.current = {
        startX: event.clientX,
        startY: event.clientY,
        startWw: ww,
        startWl: wl,
      }
      return
    }

    if (!displayUrl) {
      return
    }
    const isPanTrigger = event.button === 1 || (event.button === 0 && event.shiftKey)
    if (!isPanTrigger) {
      return
    }

    event.preventDefault()
    suppressClickRef.current = false
    const currentView = viewStateRef.current
    panDragStateRef.current = {
      startX: event.clientX,
      startY: event.clientY,
      startPanX: currentView.panX,
      startPanY: currentView.panY,
    }
    setIsPanning(true)
  }

  function handleViewportClick(event: ReactMouseEvent<HTMLDivElement>) {
    if (suppressClickRef.current) {
      suppressClickRef.current = false
      return
    }
    const rect = viewportRef.current?.getBoundingClientRect()
    if (!rect || naturalSize.width === 0) {
      return
    }

    const localX = event.clientX - rect.left
    const localY = event.clientY - rect.top
    const centerX = rect.width / 2
    const centerY = rect.height / 2
    const unpannedX = localX - viewState.panX
    const unpannedY = localY - viewState.panY
    const baseX = (unpannedX - centerX) / viewState.zoom + centerX
    const baseY = (unpannedY - centerY) / viewState.zoom + centerY

    const scale = Math.min(rect.width / naturalSize.width, rect.height / naturalSize.height)
    const rw = naturalSize.width * scale
    const rh = naturalSize.height * scale
    const imgLeft = (rect.width - rw) / 2
    const imgTop = (rect.height - rh) / 2
    const x = (baseX - imgLeft) / rw
    const y = (baseY - imgTop) / rh
    onCrosshairChange({
      x: Math.min(1, Math.max(0, x)),
      y: Math.min(1, Math.max(0, y)),
    })
  }

  const cr = contentRect ?? { left: 0, top: 0, width: 1, height: 1 }
  const chLeft = `${(cr.left + crosshair.x * cr.width) * 100}%`
  const chTop = `${(cr.top + crosshair.y * cr.height) * 100}%`
  const crLeft = `${cr.left * 100}%`
  const crTop = `${cr.top * 100}%`
  const crRight = `${(1 - cr.left - cr.width) * 100}%`
  const crBottom = `${(1 - cr.top - cr.height) * 100}%`
  const displayNaturalSize = naturalSize.width > 0 ? `${naturalSize.width}×${naturalSize.height}` : ''
  const zoomLabel = `${viewState.zoom.toFixed(2)}×`

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
        <Alert severity="warning" sx={{ m: 1 }}>
          {errorText}
        </Alert>
      ) : null}

      {!errorText ? (
        <Box
          ref={viewportRef}
          data-slice-viewport={axis}
          onClick={displayUrl ? handleViewportClick : undefined}
          onDoubleClick={displayUrl ? resetViewToFit : undefined}
          sx={{
            position: 'absolute',
            inset: 0,
            backgroundColor: '#000',
            cursor: displayUrl ? (isPanning ? 'grabbing' : 'crosshair') : 'default',
          }}
        >
          {displayUrl ? (
            <>
              <Box
                sx={{
                  position: 'absolute',
                  inset: 0,
                  transform: `translate(${viewState.panX}px, ${viewState.panY}px)`,
                  transformOrigin: 'center center',
                  pointerEvents: 'none',
                }}
              >
                <Box
                  sx={{
                    position: 'absolute',
                    inset: 0,
                    transform: `scale(${viewState.zoom})`,
                    transformOrigin: 'center center',
                  }}
                >
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

                  {contentRect ? (
                    <Box sx={{ position: 'absolute', inset: 0 }}>
                      <Box
                        sx={{
                          position: 'absolute',
                          left: chLeft,
                          top: crTop,
                          bottom: crBottom,
                          width: '1.5px',
                          transform: 'translateX(-50%)',
                          backgroundColor: accent,
                          opacity: 0.85,
                        }}
                      />
                      <Box
                        sx={{
                          position: 'absolute',
                          left: crLeft,
                          right: crRight,
                          top: chTop,
                          height: '1.5px',
                          transform: 'translateY(-50%)',
                          backgroundColor: accent,
                          opacity: 0.85,
                        }}
                      />
                    </Box>
                  ) : null}
                </Box>
              </Box>

              {isStale ? (
                <CircularProgress
                  size={20}
                  thickness={5}
                  sx={{
                    position: 'absolute',
                    top: 8,
                    right: 64,
                    color: accent,
                    opacity: 0.7,
                  }}
                />
              ) : null}
              <Button
                size="small"
                variant={isViewTransformed ? 'contained' : 'outlined'}
                onClick={(event) => {
                  event.stopPropagation()
                  resetViewToFit()
                }}
                sx={{
                  position: 'absolute',
                  top: 8,
                  right: 8,
                  minWidth: 46,
                  px: 0.9,
                  py: 0.15,
                  lineHeight: 1.2,
                  fontSize: '0.68rem',
                }}
              >
                Fit
              </Button>
              {loadError ? (
                <Alert
                  severity="warning"
                  sx={{
                    position: 'absolute',
                    top: 8,
                    left: 8,
                    right: 40,
                    py: 0.1,
                    '& .MuiAlert-message': {
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      whiteSpace: 'nowrap',
                    },
                  }}
                >
                  Slice refresh failed. Showing previous frame.
                </Alert>
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
            <Box sx={{ p: 2 }}>
              <Alert severity="error">{loadError}</Alert>
            </Box>
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
        <Typography variant="caption" sx={{ color: 'rgba(0, 255, 0, 0.95)', fontSize: '0.65rem' }}>
          Scroll slices · Ctrl+Scroll zoom · Shift+Drag/MMB pan
        </Typography>
        <Typography variant="caption" sx={{ color: 'rgba(0, 255, 0, 0.95)', fontSize: '0.65rem' }}>
          {displayNaturalSize} {displayNaturalSize ? '· ' : ''}
          {zoomLabel}
        </Typography>
      </Stack>
    </Box>
  )
}

export default SliceView
