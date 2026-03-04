import {
  Alert,
  Box,
  Button,
  Chip,
  Divider,
  Paper,
  Stack,
  Typography,
} from '@mui/material'
import { useEffect, useState } from 'react'
import { Link as RouterLink, useParams } from 'react-router-dom'

import {
  apiClient,
  getApiErrorMessage,
  type Axis,
  type SeriesInfo,
  type SliceQuery,
  type VolumeInfo,
} from '../services/api'
import BlendSlider from '../components/viewer/BlendSlider'
import LayerToggle from '../components/viewer/LayerToggle'
import OpacitySlider from '../components/viewer/OpacitySlider'
import SeriesSelector from '../components/viewer/SeriesSelector'
import SliceSlider from '../components/viewer/SliceSlider'
import SliceView from '../components/viewer/SliceView'
import Surface3DView from '../components/viewer/Surface3DView'
import ViewerGrid2x2 from '../components/viewer/ViewerGrid2x2'
import WindowLevelControl from '../components/viewer/WindowLevelControl'
import { useSliceNavigation } from '../components/viewer/useSliceNavigation'
import { useWindowLevel } from '../components/viewer/useWindowLevel'

const PANEL_ACCENTS: Record<Axis, string> = {
  axial: '#fbbf24',
  coronal: '#ef4444',
  sagittal: '#22c55e',
}

const LAYER_META: Record<number, { label: string; color: string; defaultOpacity: number }> = {
  1: { label: 'Kidney', color: '#22d3ee', defaultOpacity: 0.15 },
  2: { label: 'Tumor', color: '#facc15', defaultOpacity: 0.2 },
  3: { label: 'Cyst', color: '#e879f9', defaultOpacity: 0.15 },
}

const SURFACE_LAYER_COLORS: Record<number, string> = {
  1: LAYER_META[1].color,
  2: LAYER_META[2].color,
  3: LAYER_META[3].color,
}

function ViewerPage() {
  const { dsid = 'unknown-dataset', pid = 'unknown-patient' } = useParams<{
    dsid: string
    pid: string
  }>()
  const [selectedSeries, setSelectedSeries] = useState<SeriesInfo | null>(null)
  const [volumeRequest, setVolumeRequest] = useState<{
    seriesId: string | null
    info: VolumeInfo | null
    error: string | null
  }>({
    seriesId: null,
    info: null,
    error: null,
  })
  const [layerState, setLayerState] = useState(() => ({
    1: { visible: true, opacity: LAYER_META[1].defaultOpacity },
    2: { visible: true, opacity: LAYER_META[2].defaultOpacity },
    3: { visible: false, opacity: LAYER_META[3].defaultOpacity },
  }))
  const [surfaceBlend, setSurfaceBlend] = useState(0.75)
  const windowLevel = useWindowLevel()

  const activeVolumeInfo =
    selectedSeries && volumeRequest.seriesId === selectedSeries.series_id
      ? volumeRequest.info
      : null
  const volumeError =
    selectedSeries && volumeRequest.seriesId === selectedSeries.series_id
      ? volumeRequest.error
      : null
  const volumeLoading =
    selectedSeries !== null && volumeRequest.seriesId !== selectedSeries.series_id
  const navigation = useSliceNavigation(activeVolumeInfo?.shape ?? null)

  useEffect(() => {
    if (!selectedSeries) {
      return
    }

    let active = true

    apiClient
      .loadSeries(dsid, pid, selectedSeries.series_id)
      .then((info) => {
        if (!active) {
          return
        }
        setVolumeRequest({
          seriesId: selectedSeries.series_id,
          info,
          error: null,
        })
      })
      .catch((requestError) => {
        if (!active) {
          return
        }
        setVolumeRequest({
          seriesId: selectedSeries.series_id,
          info: null,
          error: getApiErrorMessage(requestError),
        })
      })

    return () => {
      active = false
    }
  }, [dsid, pid, selectedSeries])

  const availableLabels = activeVolumeInfo?.labels ?? []
  const visibleLayers = availableLabels.filter((label) => layerState[label as 1 | 2 | 3]?.visible)

  const sliceQuery: SliceQuery = {
    ww: windowLevel.ww,
    wl: windowLevel.wl,
    layers: visibleLayers,
    opacity_1: layerState[1].opacity,
    opacity_2: layerState[2].opacity,
    opacity_3: layerState[3].opacity,
  }

  const viewerPanels = {
    axial: {
      caption: describeSlice('axial', navigation.sliceIndices.axial, navigation.getMaxIndex('axial')),
      content: (
        <SlicePanel
          accent={PANEL_ACCENTS.axial}
          axis="axial"
          crosshair={navigation.getCrosshair('axial')}
          errorText={volumeError}
          index={navigation.sliceIndices.axial}
          maxIndex={navigation.getMaxIndex('axial')}
          onCrosshairChange={(point) => navigation.setFromPanelPosition('axial', point)}
          onSliceChange={(index) => navigation.setSlice('axial', index)}
          onWindowLevelDrag={windowLevel.applyDrag}
          query={sliceQuery}
          requestKey={activeVolumeInfo?.series_id ?? null}
          wl={windowLevel.wl}
          ww={windowLevel.ww}
        />
      ),
    },
    coronal: {
      caption: describeSlice(
        'coronal',
        navigation.sliceIndices.coronal,
        navigation.getMaxIndex('coronal'),
      ),
      content: (
        <SlicePanel
          accent={PANEL_ACCENTS.coronal}
          axis="coronal"
          crosshair={navigation.getCrosshair('coronal')}
          errorText={volumeError}
          index={navigation.sliceIndices.coronal}
          maxIndex={navigation.getMaxIndex('coronal')}
          onCrosshairChange={(point) => navigation.setFromPanelPosition('coronal', point)}
          onSliceChange={(index) => navigation.setSlice('coronal', index)}
          onWindowLevelDrag={windowLevel.applyDrag}
          query={sliceQuery}
          requestKey={activeVolumeInfo?.series_id ?? null}
          wl={windowLevel.wl}
          ww={windowLevel.ww}
        />
      ),
    },
    sagittal: {
      caption: describeSlice(
        'sagittal',
        navigation.sliceIndices.sagittal,
        navigation.getMaxIndex('sagittal'),
      ),
      content: (
        <SlicePanel
          accent={PANEL_ACCENTS.sagittal}
          axis="sagittal"
          crosshair={navigation.getCrosshair('sagittal')}
          errorText={volumeError}
          index={navigation.sliceIndices.sagittal}
          maxIndex={navigation.getMaxIndex('sagittal')}
          onCrosshairChange={(point) => navigation.setFromPanelPosition('sagittal', point)}
          onSliceChange={(index) => navigation.setSlice('sagittal', index)}
          onWindowLevelDrag={windowLevel.applyDrag}
          query={sliceQuery}
          requestKey={activeVolumeInfo?.series_id ?? null}
          wl={windowLevel.wl}
          ww={windowLevel.ww}
        />
      ),
    },
    surface: {
      caption: activeVolumeInfo?.has_mask
        ? 'Interactive segmentation surface'
        : 'No segmentation available',
      content: (
        <Stack
          spacing={0.75}
          sx={{
            height: '100%',
            p: 0.5,
            background: 'transparent',
          }}
        >
          <Surface3DView
            blend={surfaceBlend}
            errorText={volumeError}
            hasMask={Boolean(activeVolumeInfo?.has_mask)}
            labelColors={SURFACE_LAYER_COLORS}
            requestKey={activeVolumeInfo?.series_id ?? null}
            visibleLabels={visibleLayers}
          />
          <Typography variant="caption" color="text.secondary">
            Visible 3D labels: {visibleLayers.length > 0 ? visibleLayers.join(', ') : 'none'}
          </Typography>
        </Stack>
      ),
    },
  }

  return (
    <Stack spacing={3}>
      <Paper
        elevation={0}
        sx={{
          px: { xs: 3, md: 4 },
          py: { xs: 3, md: 3.5 },
        }}
      >
        <Stack spacing={3}>
          <Stack
            direction={{ xs: 'column', xl: 'row' }}
            spacing={2.5}
            justifyContent="space-between"
          >
            <Stack spacing={1.25} maxWidth={420}>
              <Typography variant="overline" color="text.secondary">
                Viewer Workspace
              </Typography>
              <Typography variant="h3">{pid}</Typography>
              <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
                <Chip label={dsid} color="primary" variant="outlined" />
                <Chip
                  label={selectedSeries?.filename ?? 'Select a series'}
                  variant="outlined"
                />
                {activeVolumeInfo ? (
                  <Chip
                    label={`${activeVolumeInfo.shape.join(' × ')}`}
                    variant="outlined"
                  />
                ) : null}
              </Stack>
            </Stack>

            <SeriesSelector
              datasetId={dsid}
              patientId={pid}
              onSeriesChange={setSelectedSeries}
            />
          </Stack>

          <Stack
            direction={{ xs: 'column', lg: 'row' }}
            spacing={1}
            alignItems={{ lg: 'center' }}
            justifyContent="space-between"
          >
            <Stack direction="row" spacing={1.5} flexWrap="wrap" useFlexGap>
              {[1, 2, 3].map((label) => (
                <LayerToggle
                  key={label}
                  checked={layerState[label as 1 | 2 | 3].visible}
                  color={LAYER_META[label].color}
                  disabled={!availableLabels.includes(label)}
                  label={LAYER_META[label].label}
                  onChange={(checked) =>
                    setLayerState((current) => ({
                      ...current,
                      [label]: {
                        ...current[label as 1 | 2 | 3],
                        visible: checked,
                      },
                    }))
                  }
                />
              ))}
            </Stack>

            <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1.5}>
              <Button
                component={RouterLink}
                to={`/datasets/${dsid}/patients`}
                variant="outlined"
              >
                Back to Patient List
              </Button>
              <Button component={RouterLink} to="/" variant="contained">
                Return Home
              </Button>
            </Stack>
          </Stack>

          {volumeLoading ? (
            <Alert severity="info">Loading volume and initial slices...</Alert>
          ) : null}
        </Stack>
      </Paper>

      <ViewerGrid2x2 panels={viewerPanels} />

      <Paper elevation={0} sx={{ px: { xs: 3, md: 4 }, py: 2.5 }}>
        <Stack
          direction={{ xs: 'column', lg: 'row' }}
          spacing={2}
          justifyContent="space-between"
          divider={
            <Divider
              orientation="vertical"
              flexItem
              sx={{ display: { xs: 'none', lg: 'block' } }}
            />
          }
        >
          <Box sx={{ flex: 1 }}>
            <WindowLevelControl
              ww={windowLevel.ww}
              wl={windowLevel.wl}
              onPreset={windowLevel.applyPreset}
            />
          </Box>
          <Stack direction="row" spacing={2} flexWrap="wrap" useFlexGap>
            {[1, 2, 3].map((label) => (
              <OpacitySlider
                key={label}
                color={LAYER_META[label].color}
                disabled={!availableLabels.includes(label) || !layerState[label as 1 | 2 | 3].visible}
                label={LAYER_META[label].label}
                onChange={(value) =>
                  setLayerState((current) => ({
                    ...current,
                    [label]: {
                      ...current[label as 1 | 2 | 3],
                      opacity: value,
                    },
                  }))
                }
                value={layerState[label as 1 | 2 | 3].opacity}
              />
            ))}
            <BlendSlider
              disabled={!activeVolumeInfo?.has_mask || visibleLayers.length === 0}
              onChange={setSurfaceBlend}
              value={surfaceBlend}
            />
          </Stack>
        </Stack>
      </Paper>
    </Stack>
  )
}

function SlicePanel({
  accent,
  axis,
  crosshair,
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
}: {
  accent: string
  axis: Axis
  crosshair: { x: number; y: number }
  errorText: string | null
  index: number
  maxIndex: number
  onCrosshairChange: (point: { x: number; y: number }) => void
  onSliceChange: (index: number) => void
  onWindowLevelDrag: (
    startWw: number,
    startWl: number,
    deltaX: number,
    deltaY: number,
  ) => void
  query: SliceQuery
  requestKey: string | null
  wl: number
  ww: number
}) {
  return (
    <Stack
      spacing={0.75}
      sx={{
        height: '100%',
        p: 0.5,
        background: 'transparent',
      }}
    >
      <SliceView
        accent={accent}
        axis={axis}
        crosshair={crosshair}
        disabled={!requestKey}
        errorText={errorText}
        index={index}
        maxIndex={maxIndex}
        onCrosshairChange={onCrosshairChange}
        onSliceChange={onSliceChange}
        onWindowLevelDrag={onWindowLevelDrag}
        query={query}
        requestKey={requestKey}
        wl={wl}
        ww={ww}
      />
      <SliceSlider
        axis={axis}
        color={accent}
        index={index}
        maxIndex={maxIndex}
        onChange={onSliceChange}
      />
    </Stack>
  )
}

function describeSlice(axis: string, index: number, maxIndex: number) {
  return `${axis.toUpperCase()} ${index + 1} / ${maxIndex + 1}`
}

export default ViewerPage
