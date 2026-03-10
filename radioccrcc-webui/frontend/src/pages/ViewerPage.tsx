import {
  Alert,
  Box,
  Button,
  Chip,
  Dialog,
  DialogActions,
  DialogContent,
  DialogContentText,
  DialogTitle,
  Divider,
  FormControl,
  InputLabel,
  MenuItem,
  Paper,
  Select,
  Stack,
  Typography,
} from '@mui/material'
import { lazy, Suspense, useEffect, useMemo, useRef, useState } from 'react'
import { Link as RouterLink, useNavigate, useParams } from 'react-router-dom'

import {
  apiClient,
  type DatasetViewerSettings,
  type PatientSummary,
  type ReviewApplyResponse,
  type ReviewOperation,
  getApiErrorMessage,
  type Axis,
  type PhaseDecision,
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
import ViewerGrid2x2 from '../components/viewer/ViewerGrid2x2'
import WindowLevelControl from '../components/viewer/WindowLevelControl'
import { useSettings } from '../hooks/useSettings'
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
const Surface3DView = lazy(() => import('../components/viewer/Surface3DView'))

function mapLayerStateFromSettings(
  settings: DatasetViewerSettings,
): {
  1: { visible: boolean; opacity: number }
  2: { visible: boolean; opacity: number }
  3: { visible: boolean; opacity: number }
} {
  return {
    1: {
      visible: settings.layers_visible.includes(1),
      opacity: settings.layers_opacity['1'] ?? LAYER_META[1].defaultOpacity,
    },
    2: {
      visible: settings.layers_visible.includes(2),
      opacity: settings.layers_opacity['2'] ?? LAYER_META[2].defaultOpacity,
    },
    3: {
      visible: settings.layers_visible.includes(3),
      opacity: settings.layers_opacity['3'] ?? LAYER_META[3].defaultOpacity,
    },
  }
}

function ViewerPage() {
  const { dsid = 'unknown-dataset', pid = 'unknown-patient' } = useParams<{
    dsid: string
    pid: string
  }>()
  const navigate = useNavigate()

  const [hydratedDatasetId, setHydratedDatasetId] = useState<string | null>(null)
  const [selectedSeries, setSelectedSeries] = useState<SeriesInfo | null>(null)
  const [preferredSeriesId, setPreferredSeriesId] = useState<string | null>(null)
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
  const [handleReloadTick, setHandleReloadTick] = useState(0)
  const [reviewDataRevision, setReviewDataRevision] = useState(0)
  const [selectedGroup, setSelectedGroup] = useState('all')
  const [phaseDecision, setPhaseDecision] = useState<PhaseDecision>('NC')
  const [pendingOperations, setPendingOperations] = useState<ReviewOperation[]>([])
  const [applyDialogOpen, setApplyDialogOpen] = useState(false)
  const [applyState, setApplyState] = useState<{
    running: boolean
    error: string | null
    response: ReviewApplyResponse | null
  }>({
    running: false,
    error: null,
    response: null,
  })
  const [patientRequest, setPatientRequest] = useState<{
    datasetId: string | null
    patients: PatientSummary[]
    error: string | null
  }>({
    datasetId: null,
    patients: [],
    error: null,
  })

  const handleRecoveryRequestedRef = useRef(false)
  const windowLevel = useWindowLevel()
  const settingsState = useSettings({
    datasetId: dsid,
    onLoadedDatasetSettings: (settings) => {
      setHydratedDatasetId(dsid)
      setPreferredSeriesId(settings.last_series)
      windowLevel.setWindowLevel(settings.ww ?? 400, settings.wl ?? 50)
      setLayerState(mapLayerStateFromSettings(settings))
    },
  })
  const {
    loadError: settingsLoadError,
    loading: settingsLoading,
    saveError: settingsSaveError,
    updateDatasetSettings,
  } = settingsState

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
  const activeLoadHandle = activeVolumeInfo?.load_handle ?? null
  const canPersistSettings =
    !settingsLoading &&
    (hydratedDatasetId === dsid || Boolean(settingsLoadError))

  const patientList = patientRequest.datasetId === dsid ? patientRequest.patients : []
  const patientLoading = patientRequest.datasetId !== dsid
  const patientError = patientRequest.datasetId === dsid ? patientRequest.error : null

  const groupOptions = useMemo(() => {
    const discovered = Array.from(
      new Set(patientList.map((patient) => normalizePatientGroup(patient.group))),
    ).sort((left, right) =>
      left.localeCompare(right, undefined, { sensitivity: 'base', numeric: true }),
    )
    return ['all', ...discovered]
  }, [patientList])

  const filteredPatients = useMemo(() => {
    return patientList
      .filter((patient) =>
        selectedGroup === 'all' ? true : normalizePatientGroup(patient.group) === selectedGroup,
      )
      .sort((left, right) =>
        left.patient_id.localeCompare(right.patient_id, undefined, {
          sensitivity: 'base',
          numeric: true,
        }),
      )
  }, [patientList, selectedGroup])

  const currentPatientIndex = filteredPatients.findIndex((patient) => patient.patient_id === pid)
  const nextPatient = currentPatientIndex >= 0 ? filteredPatients[currentPatientIndex + 1] ?? null : null
  const canLoadNextPatient = nextPatient !== null

  const queuedReclassifications = pendingOperations.filter((entry) => entry.action === 'reclassify').length
  const queuedDeletes = pendingOperations.filter((entry) => entry.action === 'delete').length

  function requestHandleReload() {
    if (!selectedSeries || handleRecoveryRequestedRef.current) {
      return
    }
    handleRecoveryRequestedRef.current = true
    setHandleReloadTick((current) => current + 1)
  }

  useEffect(() => {
    let active = true

    apiClient
      .listPatients(dsid)
      .then((patients) => {
        if (!active) {
          return
        }
        setPatientRequest({
          datasetId: dsid,
          patients,
          error: null,
        })
      })
      .catch((requestError) => {
        if (!active) {
          return
        }
        setPatientRequest({
          datasetId: dsid,
          patients: [],
          error: getApiErrorMessage(requestError),
        })
      })

    return () => {
      active = false
    }
  }, [dsid, reviewDataRevision])

  useEffect(() => {
    if (selectedGroup === 'all') {
      return
    }
    if (!groupOptions.includes(selectedGroup)) {
      setSelectedGroup('all')
    }
  }, [groupOptions, selectedGroup])

  useEffect(() => {
    if (patientLoading || filteredPatients.length === 0) {
      return
    }
    if (currentPatientIndex >= 0) {
      return
    }
    navigate(`/datasets/${dsid}/patients/${filteredPatients[0].patient_id}/viewer`)
  }, [currentPatientIndex, dsid, filteredPatients, navigate, patientLoading])

  useEffect(() => {
    setPendingOperations([])
    setApplyDialogOpen(false)
  }, [pid])

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
        handleRecoveryRequestedRef.current = false
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
        handleRecoveryRequestedRef.current = false
        setVolumeRequest({
          seriesId: selectedSeries.series_id,
          info: null,
          error: getApiErrorMessage(requestError),
        })
      })

    return () => {
      active = false
    }
  }, [dsid, handleReloadTick, pid, selectedSeries])

  const availableLabels = activeVolumeInfo?.labels ?? []
  const visibleLayers = availableLabels.filter((label) => layerState[label as 1 | 2 | 3]?.visible)
  const persistedVisibleLayers = useMemo(
    () => ([1, 2, 3] as const).filter((label) => layerState[label].visible),
    [layerState],
  )

  const sliceQuery: SliceQuery = {
    load_handle: activeLoadHandle ?? undefined,
    ww: windowLevel.ww,
    wl: windowLevel.wl,
    layers: visibleLayers,
    opacity_1: layerState[1].opacity,
    opacity_2: layerState[2].opacity,
    opacity_3: layerState[3].opacity,
  }

  useEffect(() => {
    if (!canPersistSettings) {
      return
    }

    updateDatasetSettings((current) => ({
      ...current,
      last_patient: pid,
      last_series: selectedSeries?.series_id ?? null,
      ww: windowLevel.ww,
      wl: windowLevel.wl,
      layers_visible: persistedVisibleLayers,
      layers_opacity: {
        1: layerState[1].opacity,
        2: layerState[2].opacity,
        3: layerState[3].opacity,
      },
    }))
  }, [
    layerState,
    persistedVisibleLayers,
    pid,
    selectedSeries?.series_id,
    canPersistSettings,
    dsid,
    hydratedDatasetId,
    updateDatasetSettings,
    windowLevel.wl,
    windowLevel.ww,
  ])

  function upsertPendingOperation(operation: ReviewOperation) {
    setPendingOperations((current) => {
      const withoutCurrentSeries = current.filter(
        (entry) =>
          !(
            entry.series_id === operation.series_id &&
            entry.action === operation.action
          ),
      )
      return [...withoutCurrentSeries, operation]
    })
  }

  function queueReclassification() {
    if (!selectedSeries) {
      return
    }
    upsertPendingOperation({
      patient_id: pid,
      series_id: selectedSeries.series_id,
      action: 'reclassify',
      target_phase: phaseDecision,
    })
  }

  function queueDelete() {
    if (!selectedSeries) {
      return
    }
    upsertPendingOperation({
      patient_id: pid,
      series_id: selectedSeries.series_id,
      action: 'delete',
    })
  }

  async function applyPendingOperations() {
    if (pendingOperations.length === 0 || applyState.running) {
      return
    }
    setApplyState({ running: true, error: null, response: null })
    try {
      const response = await apiClient.applyReviewOperations(dsid, {
        operations: pendingOperations,
      })
      setApplyState({
        running: false,
        error: null,
        response,
      })
      setPendingOperations([])
      setReviewDataRevision((current) => current + 1)
      setPreferredSeriesId(selectedSeries?.series_id ?? null)
      setHandleReloadTick((current) => current + 1)
    } catch (requestError) {
      setApplyState({
        running: false,
        error: getApiErrorMessage(requestError),
        response: null,
      })
    } finally {
      setApplyDialogOpen(false)
    }
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
          onHandleExpired={requestHandleReload}
          onWindowLevelDrag={windowLevel.applyDrag}
          query={sliceQuery}
          requestKey={activeLoadHandle}
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
          onHandleExpired={requestHandleReload}
          onWindowLevelDrag={windowLevel.applyDrag}
          query={sliceQuery}
          requestKey={activeLoadHandle}
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
          onHandleExpired={requestHandleReload}
          onWindowLevelDrag={windowLevel.applyDrag}
          query={sliceQuery}
          requestKey={activeLoadHandle}
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
          <Suspense
            fallback={
              <Stack
                spacing={1}
                alignItems="center"
                justifyContent="center"
                sx={{ flex: 1 }}
              >
                <Typography variant="body2" color="text.secondary">
                  Loading 3D renderer...
                </Typography>
              </Stack>
            }
          >
            <Surface3DView
              blend={surfaceBlend}
              errorText={volumeError}
              hasMask={Boolean(activeVolumeInfo?.has_mask)}
              labelColors={SURFACE_LAYER_COLORS}
              loadHandle={activeLoadHandle}
              onHandleExpired={requestHandleReload}
              visibleLabels={visibleLayers}
            />
          </Suspense>
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
            <Stack spacing={1.25} maxWidth={500}>
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
                {queuedReclassifications + queuedDeletes > 0 ? (
                  <Chip
                    color="secondary"
                    label={`Pending: ${queuedReclassifications + queuedDeletes}`}
                    variant="outlined"
                  />
                ) : null}
              </Stack>
            </Stack>

            <Stack spacing={1.5} minWidth={{ xs: '100%', xl: 460 }}>
              <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1.5}>
                <FormControl fullWidth>
                  <InputLabel id="viewer-group-filter-label">Group</InputLabel>
                  <Select
                    labelId="viewer-group-filter-label"
                    label="Group"
                    value={selectedGroup}
                    onChange={(event) => setSelectedGroup(event.target.value)}
                    disabled={patientLoading || groupOptions.length === 0}
                  >
                    {groupOptions.map((group) => (
                      <MenuItem key={group} value={group}>
                        {group === 'all' ? 'All' : group}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
                <FormControl fullWidth>
                  <InputLabel id="viewer-patient-selector-label">Patient</InputLabel>
                  <Select
                    labelId="viewer-patient-selector-label"
                    label="Patient"
                    value={filteredPatients.some((patient) => patient.patient_id === pid) ? pid : ''}
                    onChange={(event) =>
                      navigate(`/datasets/${dsid}/patients/${event.target.value}/viewer`)
                    }
                    disabled={patientLoading || filteredPatients.length === 0}
                  >
                    {filteredPatients.map((patient) => (
                      <MenuItem key={patient.patient_id} value={patient.patient_id}>
                        {patient.patient_id}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
                <Button
                  variant="outlined"
                  onClick={() => {
                    if (nextPatient) {
                      navigate(`/datasets/${dsid}/patients/${nextPatient.patient_id}/viewer`)
                    }
                  }}
                  disabled={!canLoadNextPatient || patientLoading}
                >
                  Next
                </Button>
              </Stack>

              <SeriesSelector
                datasetId={dsid}
                patientId={pid}
                onSeriesChange={setSelectedSeries}
                preferredSeriesId={preferredSeriesId}
                reloadKey={reviewDataRevision}
              />
            </Stack>
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

          <Stack direction={{ xs: 'column', xl: 'row' }} spacing={1.5} alignItems={{ xl: 'center' }}>
            <FormControl sx={{ minWidth: 160 }}>
              <InputLabel id="phase-decision-label">Phase</InputLabel>
              <Select
                labelId="phase-decision-label"
                label="Phase"
                value={phaseDecision}
                onChange={(event) => setPhaseDecision(event.target.value as PhaseDecision)}
              >
                <MenuItem value="NC">NC</MenuItem>
                <MenuItem value="ART">ART</MenuItem>
                <MenuItem value="VEN">VEN</MenuItem>
              </Select>
            </FormControl>
            <Button
              variant="outlined"
              onClick={queueReclassification}
              disabled={!selectedSeries}
            >
              Queue Reclassify
            </Button>
            <Button
              variant="outlined"
              color="error"
              onClick={queueDelete}
              disabled={!selectedSeries}
            >
              Queue Delete
            </Button>
            <Button
              variant="text"
              onClick={() => setPendingOperations([])}
              disabled={pendingOperations.length === 0}
            >
              Clear Pending
            </Button>
            <Button
              variant="contained"
              color="warning"
              onClick={() => setApplyDialogOpen(true)}
              disabled={pendingOperations.length === 0 || applyState.running}
            >
              Apply Changes
            </Button>
          </Stack>

          {pendingOperations.length > 0 ? (
            <Alert severity="info">
              Pending operations: {queuedReclassifications} reclassify, {queuedDeletes} delete.
            </Alert>
          ) : null}

          {volumeLoading ? (
            <Alert severity="info">Loading volume and initial slices...</Alert>
          ) : null}
          {patientError ? (
            <Alert severity="warning">Failed to load patient list: {patientError}</Alert>
          ) : null}
          {settingsLoadError ? (
            <Alert severity="warning">
              Failed to load saved viewer settings: {settingsLoadError}
            </Alert>
          ) : null}
          {settingsSaveError ? (
            <Alert severity="warning">
              Failed to persist viewer settings: {settingsSaveError}
            </Alert>
          ) : null}
          {applyState.error ? (
            <Alert severity="error">Apply failed: {applyState.error}</Alert>
          ) : null}
          {applyState.response ? (
            <Alert severity={applyState.response.summary.failed > 0 ? 'warning' : 'success'}>
              Batch {applyState.response.batch_id}: {applyState.response.summary.applied} applied,
              {' '}
              {applyState.response.summary.skipped} skipped, {applyState.response.summary.failed} failed.
            </Alert>
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

      <Dialog open={applyDialogOpen} onClose={() => setApplyDialogOpen(false)}>
        <DialogTitle>Apply Review Changes</DialogTitle>
        <DialogContent>
          <DialogContentText>
            This will execute {pendingOperations.length} queued operation(s) for patient {pid}.
            Reclassification updates manifest metadata; delete actions move files to recycle paths.
          </DialogContentText>
          <DialogContentText sx={{ mt: 1 }}>
            Queue summary: {queuedReclassifications} reclassify, {queuedDeletes} delete.
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setApplyDialogOpen(false)} disabled={applyState.running}>
            Cancel
          </Button>
          <Button
            onClick={applyPendingOperations}
            color="warning"
            variant="contained"
            disabled={applyState.running}
          >
            {applyState.running ? 'Applying...' : 'Apply Changes'}
          </Button>
        </DialogActions>
      </Dialog>
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
  onHandleExpired,
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
  onHandleExpired: () => void
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
        onHandleExpired={onHandleExpired}
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

function normalizePatientGroup(group: string | null): string {
  const normalized = group?.trim()
  if (!normalized) {
    return 'Unknown'
  }
  return normalized
}

function describeSlice(axis: string, index: number, maxIndex: number) {
  return `${axis.toUpperCase()} ${index + 1} / ${maxIndex + 1}`
}

export default ViewerPage
