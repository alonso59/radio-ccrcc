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
  const [requestState, setRequestState] = useState<{
    key: string | null
    url: string | null
    error: string | null
  }>({
    key: null,
    url: null,
    error: null,
  })
  const [naturalSize, setNaturalSize] = useState<{
    key: string | null
    width: number
    height: number
  }>({
    key: null,
    width: 0,
    height: 0,
  })
  const dragStateRef = useRef<{
    startX: number
    startY: number
    startWw: number
    startWl: number
  } | null>(null)
  const wrapperRef = useRef<HTMLDivElement | null>(null)

  const queryKey = JSON.stringify(query)
  const fetchKey = !disabled && requestKey ? `${requestKey}:${axis}:${index}:${queryKey}` : null
  const isLoading = Boolean(fetchKey) && requestState.key !== fetchKey
  const imageUrl = requestState.key === fetchKey ? requestState.url : null
  const loadError = requestState.key === fetchKey ? requestState.error : null

  useEffect(() => {
    if (!fetchKey) {
      return
    }

    let active = true
    apiClient
      .getSliceBlob(axis, index, query)
      .then((blob) => {
        if (!active) {
          return
        }
        const nextUrl = URL.createObjectURL(blob)
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
        setRequestState({
          key: fetchKey,
          url: null,
          error: getApiErrorMessage(requestError),
        })
      })

    return () => {
      active = false
    }
  }, [axis, fetchKey, index, query])

  useEffect(() => {
    return () => {
      if (requestState.url) {
        URL.revokeObjectURL(requestState.url)
      }
    }
  }, [requestState.url])

  useEffect(() => {
    function handleMouseMove(event: MouseEvent) {
      const dragState = dragStateRef.current
      if (!dragState) {
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
    }

    window.addEventListener('mousemove', handleMouseMove)
    window.addEventListener('mouseup', handleMouseUp)

    return () => {
      window.removeEventListener('mousemove', handleMouseMove)
      window.removeEventListener('mouseup', handleMouseUp)
    }
  }, [onWindowLevelDrag])

  function handleWheel(event: ReactWheelEvent<HTMLDivElement>) {
    event.preventDefault()
    if (disabled || maxIndex <= 0) {
      return
    }

    const delta = event.deltaY > 0 ? 1 : -1
    const nextIndex = Math.min(maxIndex, Math.max(0, index + delta))
    if (nextIndex !== index) {
      onSliceChange(nextIndex)
    }
  }

  function handleMouseDown(event: ReactMouseEvent<HTMLDivElement>) {
    if (event.button !== 2) {
      return
    }

    event.preventDefault()
    dragStateRef.current = {
      startX: event.clientX,
      startY: event.clientY,
      startWw: ww,
      startWl: wl,
    }
  }

  const overlayX = `${crosshair.x * 100}%`
  const overlayY = `${crosshair.y * 100}%`

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
        borderRadius: 3,
        backgroundColor: '#090909',
        border: '1px solid',
        borderColor: 'divider',
        overflow: 'hidden',
      }}
    >
      {errorText ? (
        <Alert severity="warning" sx={{ m: 2 }}>
          {errorText}
        </Alert>
      ) : null}

      {!errorText ? (
        <Box
          sx={{
            position: 'absolute',
            inset: 0,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            p: 2,
          }}
        >
          <Box
            data-slice-viewport={axis}
            sx={{
              width: 'auto',
              height: '100%',
              maxWidth: '100%',
              aspectRatio: '1 / 1',
              borderRadius: 3,
              overflow: 'hidden',
              backgroundColor: '#000',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            {imageUrl ? (
              <Box
                ref={wrapperRef}
                data-slice-axis={axis}
                data-slice-index={index}
                data-crosshair-x={crosshair.x.toFixed(4)}
                data-crosshair-y={crosshair.y.toFixed(4)}
                onClick={(event) => {
                  const rect = event.currentTarget.getBoundingClientRect()
                  const x = (event.clientX - rect.left) / rect.width
                  const y = (event.clientY - rect.top) / rect.height
                  onCrosshairChange({
                    x: Math.min(1, Math.max(0, x)),
                    y: Math.min(1, Math.max(0, y)),
                  })
                }}
                sx={{
                  position: 'relative',
                  display: 'inline-flex',
                  maxWidth: '100%',
                  maxHeight: '100%',
                  alignItems: 'center',
                  justifyContent: 'center',
                  cursor: 'crosshair',
                }}
              >
                <Box
                  component="img"
                  src={imageUrl}
                  alt={`${axis} slice ${index + 1}`}
                  onLoad={(event: SyntheticEvent<HTMLImageElement>) => {
                    const target = event.currentTarget
                    setNaturalSize({
                      key: fetchKey,
                      width: target.naturalWidth,
                      height: target.naturalHeight,
                    })
                  }}
                  sx={{
                    display: 'block',
                    maxWidth: '100%',
                    maxHeight: '100%',
                    width: 'auto',
                    height: 'auto',
                    userSelect: 'none',
                  }}
                />
                <Box
                  sx={{
                    position: 'absolute',
                    inset: 0,
                    pointerEvents: 'none',
                  }}
                >
                  <Box
                    data-crosshair-line="vertical"
                    sx={{
                      position: 'absolute',
                      left: overlayX,
                      top: 0,
                      bottom: 0,
                      width: 2,
                      transform: 'translateX(-50%)',
                      backgroundColor: accent,
                      opacity: 0.95,
                      boxShadow: `0 0 10px ${accent}`,
                    }}
                  />
                  <Box
                    data-crosshair-line="horizontal"
                    sx={{
                      position: 'absolute',
                      left: 0,
                      right: 0,
                      top: overlayY,
                      height: 2,
                      transform: 'translateY(-50%)',
                      backgroundColor: accent,
                      opacity: 0.95,
                      boxShadow: `0 0 10px ${accent}`,
                    }}
                  />
                </Box>
              </Box>
            ) : isLoading ? (
              <Stack spacing={1} alignItems="center">
                <CircularProgress size={28} />
                <Typography variant="body2" color="text.secondary">
                  Fetching {axis} slice...
                </Typography>
              </Stack>
            ) : loadError ? (
              <Alert severity="error">{loadError}</Alert>
            ) : (
              <Stack spacing={1} alignItems="center">
                <Typography variant="body2" color="text.secondary">
                  Select a series to load {axis} slices.
                </Typography>
              </Stack>
            )}
          </Box>
        </Box>
      ) : null}

      <Stack
        direction="row"
        spacing={1}
        justifyContent="space-between"
        sx={{
          position: 'absolute',
          left: 12,
          right: 12,
          bottom: 10,
          pointerEvents: 'none',
        }}
      >
        <Typography variant="caption" color="text.secondary">
          Scroll to navigate
        </Typography>
        <Typography variant="caption" color="text.secondary">
          {naturalSize.key === fetchKey ? `${naturalSize.width}×${naturalSize.height}` : ''}
        </Typography>
      </Stack>
    </Box>
  )
}

export default SliceView
