import { useEffect, useState } from 'react'
import { Alert, Box, CircularProgress, Stack, Typography } from '@mui/material'
import { Canvas } from '@react-three/fiber'
import { Bounds, OrbitControls } from '@react-three/drei'
import axios from 'axios'
import {
  Mesh,
  MeshStandardMaterial,
  Object3D,
  type Material,
} from 'three'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'

import { apiClient, getApiErrorMessage } from '../../services/api'

interface Surface3DViewProps {
  blend: number
  errorText?: string | null
  hasMask: boolean
  labelColors: Record<number, string>
  requestKey: string | null
  visibleLabels: number[]
}

interface MeshEntry {
  label: number
  object: Object3D
}

function isMeshObject(node: Object3D): node is Mesh {
  return 'isMesh' in node && Boolean((node as { isMesh?: boolean }).isMesh)
}

function toMaterialList(material: Material | Material[]): Material[] {
  return Array.isArray(material) ? material : [material]
}

function disposeObject(root: Object3D): void {
  root.traverse((node) => {
    if (!isMeshObject(node)) {
      return
    }
    node.geometry.dispose()
    toMaterialList(node.material).forEach((material) => material.dispose())
  })
}

function applyVisualStyle(root: Object3D, color: string, blend: number): void {
  root.traverse((node) => {
    if (!isMeshObject(node)) {
      return
    }

    toMaterialList(node.material).forEach((material) => material.dispose())
    node.material = new MeshStandardMaterial({
      color,
      opacity: blend,
      transparent: blend < 1,
      roughness: 0.65,
      metalness: 0.05,
    })
    node.castShadow = false
    node.receiveShadow = true
  })
}

function updateBlend(root: Object3D, blend: number): void {
  root.traverse((node) => {
    if (!isMeshObject(node)) {
      return
    }
    toMaterialList(node.material).forEach((material) => {
      if (material instanceof MeshStandardMaterial) {
        material.opacity = blend
        material.transparent = blend < 1
        material.needsUpdate = true
      }
    })
  })
}

async function loadMesh(label: number, color: string, blend: number): Promise<MeshEntry | null> {
  try {
    const blob = await apiClient.getMeshBlob(label)
    const objectUrl = URL.createObjectURL(blob)
    try {
      const loader = new GLTFLoader()
      const gltf = await loader.loadAsync(objectUrl)
      const object = gltf.scene.clone(true)
      applyVisualStyle(object, color, blend)
      return { label, object }
    } finally {
      URL.revokeObjectURL(objectUrl)
    }
  } catch (error) {
    if (axios.isAxiosError(error) && error.response?.status === 404) {
      return null
    }
    throw error
  }
}

function Surface3DView({
  blend,
  errorText,
  hasMask,
  labelColors,
  requestKey,
  visibleLabels,
}: Surface3DViewProps) {
  const [requestState, setRequestState] = useState<{
    error: string | null
    key: string | null
    meshes: MeshEntry[]
    seriesKey: string | null
  }>({
    error: null,
    key: null,
    meshes: [],
    seriesKey: null,
  })

  const sortedVisibleLabels = [...visibleLabels].sort((left, right) => left - right)
  const visibleKey = sortedVisibleLabels.join(',')
  const seriesKey = requestKey && hasMask ? requestKey : null
  const fetchKey = seriesKey && visibleKey ? `${seriesKey}::${visibleKey}` : null

  useEffect(() => {
    if (!fetchKey || !seriesKey) {
      return
    }

    let active = true
    const labelsToLoad = visibleKey
      .split(',')
      .map((value) => Number(value))
      .filter((value) => !Number.isNaN(value))

    Promise.all(
      labelsToLoad.map((label) => {
        const color = labelColors[label] ?? '#f5f5f5'
        return loadMesh(label, color, 1)
      }),
    )
      .then((entries) => {
        if (!active) {
          entries.forEach((entry) => {
            if (entry) {
              disposeObject(entry.object)
            }
          })
          return
        }

        const loadedEntries = entries.filter((entry): entry is MeshEntry => entry !== null)
        setRequestState((current) => {
          current.meshes.forEach((meshEntry) => disposeObject(meshEntry.object))
          return {
            error: null,
            key: fetchKey,
            meshes: loadedEntries,
            seriesKey,
          }
        })
      })
      .catch((requestError) => {
        if (!active) {
          return
        }
        setRequestState((current) => ({
          error: getApiErrorMessage(requestError),
          key: fetchKey,
          meshes: current.seriesKey === seriesKey ? current.meshes : [],
          seriesKey,
        }))
      })

    return () => {
      active = false
    }
  }, [fetchKey, labelColors, seriesKey, visibleKey])

  useEffect(() => {
    requestState.meshes.forEach((entry) => updateBlend(entry.object, blend))
  }, [blend, requestState.meshes])

  useEffect(() => {
    if (!requestState.seriesKey || requestState.seriesKey === seriesKey) {
      return
    }
    requestState.meshes.forEach((entry) => disposeObject(entry.object))
  }, [requestState.meshes, requestState.seriesKey, seriesKey])

  useEffect(() => {
    return () => {
      requestState.meshes.forEach((entry) => disposeObject(entry.object))
    }
  }, [requestState.meshes])

  if (errorText) {
    return (
      <Alert severity="warning" sx={{ m: 2 }}>
        {errorText}
      </Alert>
    )
  }

  if (!requestKey) {
    return (
      <SurfacePanelMessage
        title="Select a series"
        description="Choose a series to initialize 3D surface rendering."
      />
    )
  }

  if (!hasMask) {
    return (
      <SurfacePanelMessage
        title="No segmentation available"
        description="This series does not include a segmentation mask for 3D surfaces."
      />
    )
  }

  if (sortedVisibleLabels.length === 0) {
    return (
      <SurfacePanelMessage
        title="All layers hidden"
        description="Enable at least one layer to render 3D surfaces."
      />
    )
  }

  const renderedMeshes = (requestState.seriesKey === seriesKey ? requestState.meshes : []).filter(
    (entry) => sortedVisibleLabels.includes(entry.label),
  )
  const loadedLabels = renderedMeshes.map((entry) => entry.label).join(',')
  const isLoading = Boolean(fetchKey) && requestState.key !== fetchKey
  const loadError = requestState.key === fetchKey ? requestState.error : null

  return (
    <Box
      data-surface-3d="panel"
      data-surface-blend={blend.toFixed(2)}
      data-surface-loaded-labels={loadedLabels}
      data-surface-loading={isLoading ? 'true' : 'false'}
      sx={{
        position: 'relative',
        flex: 1,
        minHeight: 0,
        borderRadius: 3,
        overflow: 'hidden',
        border: '1px solid',
        borderColor: 'divider',
        backgroundColor: '#050505',
      }}
    >
      {loadError && renderedMeshes.length === 0 ? (
        <Alert severity="error" sx={{ m: 2 }}>
          {loadError}
        </Alert>
      ) : (
        <Canvas
          camera={{
            far: 5000,
            fov: 45,
            near: 0.1,
            position: [300, 260, 260],
          }}
        >
          <color attach="background" args={['#050505']} />
          <ambientLight intensity={0.7} />
          <directionalLight position={[360, 440, 240]} intensity={1.2} />
          <directionalLight position={[-220, -180, -180]} intensity={0.45} />
          <Bounds fit clip observe margin={1.25}>
            <group>
              {renderedMeshes.map((entry) => (
                <primitive key={entry.label} object={entry.object} />
              ))}
            </group>
          </Bounds>
          <OrbitControls enableDamping dampingFactor={0.09} />
        </Canvas>
      )}

      {isLoading ? (
        <Stack
          spacing={1}
          alignItems="center"
          justifyContent="center"
          sx={{
            position: 'absolute',
            inset: 0,
            backgroundColor: 'rgba(0, 0, 0, 0.25)',
            pointerEvents: 'none',
          }}
        >
          <CircularProgress size={26} />
          <Typography variant="caption" color="text.secondary">
            Generating 3D surface...
          </Typography>
        </Stack>
      ) : null}
    </Box>
  )
}

function SurfacePanelMessage({
  description,
  title,
}: {
  description: string
  title: string
}) {
  return (
    <Box
      sx={{
        flex: 1,
        minHeight: 0,
        borderRadius: 3,
        border: '1px dashed',
        borderColor: 'divider',
        background:
          'radial-gradient(circle at top, rgba(255, 255, 255, 0.08), transparent 42%), rgba(255,255,255,0.015)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        textAlign: 'center',
        px: 3,
      }}
    >
      <Stack spacing={1.25} alignItems="center" maxWidth={420}>
        <Typography variant="h6">{title}</Typography>
        <Typography variant="body2" color="text.secondary">
          {description}
        </Typography>
      </Stack>
    </Box>
  )
}

export default Surface3DView
