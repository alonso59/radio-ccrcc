import { useState } from 'react'

import type { Axis } from '../../services/api'

interface VolumePoint {
  x: number
  y: number
  z: number
}

interface CrosshairPoint {
  x: number
  y: number
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value))
}

function midpoint(size: number): number {
  return Math.max(0, Math.floor(size / 2))
}

function normalize(index: number, size: number): number {
  if (size <= 1) {
    return 0.5
  }
  return index / (size - 1)
}

function fromFraction(fraction: number, size: number): number {
  if (size <= 1) {
    return 0
  }
  return clamp(Math.round(fraction * (size - 1)), 0, size - 1)
}

function defaultPoint(shape: number[] | null): VolumePoint {
  if (!shape || shape.length < 3) {
    return { x: 0, y: 0, z: 0 }
  }

  return {
    x: midpoint(shape[0]),
    y: midpoint(shape[1]),
    z: midpoint(shape[2]),
  }
}

function clampPoint(point: VolumePoint, shape: number[]): VolumePoint {
  return {
    x: clamp(point.x, 0, Math.max(0, shape[0] - 1)),
    y: clamp(point.y, 0, Math.max(0, shape[1] - 1)),
    z: clamp(point.z, 0, Math.max(0, shape[2] - 1)),
  }
}

export function useSliceNavigation(shape: number[] | null) {
  const shapeKey = shape?.join('x') ?? null
  const [state, setState] = useState<{
    shapeKey: string | null
    point: VolumePoint
  }>({
    shapeKey: null,
    point: { x: 0, y: 0, z: 0 },
  })

  const currentPoint =
    shape && state.shapeKey === shapeKey ? clampPoint(state.point, shape) : defaultPoint(shape)

  const sliceIndices = {
    axial: currentPoint.z,
    coronal: currentPoint.y,
    sagittal: currentPoint.x,
  }

  function commit(nextPoint: VolumePoint) {
    if (!shape || !shapeKey) {
      return
    }

    setState({
      shapeKey,
      point: clampPoint(nextPoint, shape),
    })
  }

  function setSlice(axis: Axis, index: number) {
    if (!shape) {
      return
    }

    if (axis === 'axial') {
      commit({ ...currentPoint, z: index })
      return
    }
    if (axis === 'coronal') {
      commit({ ...currentPoint, y: index })
      return
    }
    commit({ ...currentPoint, x: index })
  }

  function setFromPanelPosition(axis: Axis, point: CrosshairPoint) {
    if (!shape) {
      return
    }

    const xFraction = clamp(point.x, 0, 1)
    const yFraction = clamp(point.y, 0, 1)

    if (axis === 'axial') {
      commit({
        x: fromFraction(xFraction, shape[0]),
        y: fromFraction(1 - yFraction, shape[1]),
        z: currentPoint.z,
      })
      return
    }

    if (axis === 'coronal') {
      commit({
        x: fromFraction(xFraction, shape[0]),
        y: currentPoint.y,
        z: fromFraction(1 - yFraction, shape[2]),
      })
      return
    }

    commit({
      x: currentPoint.x,
      y: fromFraction(xFraction, shape[1]),
      z: fromFraction(1 - yFraction, shape[2]),
    })
  }

  function getCrosshair(axis: Axis): CrosshairPoint {
    if (!shape) {
      return { x: 0.5, y: 0.5 }
    }

    if (axis === 'axial') {
      return {
        x: normalize(currentPoint.x, shape[0]),
        y: normalize(shape[1] - 1 - currentPoint.y, shape[1]),
      }
    }

    if (axis === 'coronal') {
      return {
        x: normalize(currentPoint.x, shape[0]),
        y: normalize(shape[2] - 1 - currentPoint.z, shape[2]),
      }
    }

    return {
      x: normalize(currentPoint.y, shape[1]),
      y: normalize(shape[2] - 1 - currentPoint.z, shape[2]),
    }
  }

  function getMaxIndex(axis: Axis): number {
    if (!shape) {
      return 0
    }
    if (axis === 'axial') {
      return Math.max(0, shape[2] - 1)
    }
    if (axis === 'coronal') {
      return Math.max(0, shape[1] - 1)
    }
    return Math.max(0, shape[0] - 1)
  }

  return {
    crosshairPoint: currentPoint,
    getCrosshair,
    getMaxIndex,
    setFromPanelPosition,
    setSlice,
    sliceIndices,
  }
}
