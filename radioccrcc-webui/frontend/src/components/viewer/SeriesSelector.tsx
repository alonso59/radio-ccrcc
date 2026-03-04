import { useEffect, useState } from 'react'
import {
  Alert,
  Box,
  Chip,
  CircularProgress,
  FormControl,
  InputLabel,
  MenuItem,
  Select,
  Stack,
  Typography,
} from '@mui/material'

import {
  apiClient,
  type SeriesInfo,
  getApiErrorMessage,
} from '../../services/api'

interface SeriesSelectorProps {
  datasetId: string
  patientId: string
  onSeriesChange?: (series: SeriesInfo | null) => void
}

function SeriesSelector({
  datasetId,
  patientId,
  onSeriesChange,
}: SeriesSelectorProps) {
  const [requestState, setRequestState] = useState<{
    scope: string | null
    seriesList: SeriesInfo[]
    error: string | null
  }>({
    scope: null,
    seriesList: [],
    error: null,
  })
  const [selectedSeriesId, setSelectedSeriesId] = useState('')
  const scope = `${datasetId}:${patientId}`
  const loading = requestState.scope !== scope
  const seriesList = requestState.scope === scope ? requestState.seriesList : []
  const error = requestState.scope === scope ? requestState.error : null

  useEffect(() => {
    let active = true

    apiClient
      .listSeries(datasetId, patientId)
      .then((series) => {
        if (!active) {
          return
        }

        setRequestState({
          scope,
          seriesList: series,
          error: null,
        })

        const defaultSeries = series[0] ?? null
        setSelectedSeriesId(defaultSeries?.series_id ?? '')
        onSeriesChange?.(defaultSeries)
      })
      .catch((requestError) => {
        if (!active) {
          return
        }

        setRequestState({
          scope,
          seriesList: [],
          error: getApiErrorMessage(requestError),
        })
        setSelectedSeriesId('')
        onSeriesChange?.(null)
      })

    return () => {
      active = false
    }
  }, [datasetId, onSeriesChange, patientId, scope])

  const selectedSeries =
    seriesList.find((series) => series.series_id === selectedSeriesId) ?? null

  return (
    <Stack spacing={1.25} minWidth={{ xs: '100%', md: 360 }}>
      <FormControl fullWidth disabled={loading || seriesList.length === 0}>
        <InputLabel id="series-selector-label">Series</InputLabel>
        <Select
          labelId="series-selector-label"
          label="Series"
          value={selectedSeriesId}
          onChange={(event) => {
            const nextId = event.target.value
            setSelectedSeriesId(nextId)
            onSeriesChange?.(
              seriesList.find((series) => series.series_id === nextId) ?? null,
            )
          }}
        >
          {seriesList.map((series) => (
            <MenuItem key={series.series_id} value={series.series_id}>
              <Stack
                direction={{ xs: 'column', sm: 'row' }}
                spacing={{ xs: 0.5, sm: 1.25 }}
                alignItems={{ sm: 'center' }}
              >
                <Typography fontWeight={600}>{series.filename}</Typography>
                <Typography variant="body2" color="text.secondary">
                  {series.type.toUpperCase()}
                </Typography>
              </Stack>
            </MenuItem>
          ))}
        </Select>
      </FormControl>

      {loading ? (
        <Stack direction="row" spacing={1} alignItems="center">
          <CircularProgress size={18} />
          <Typography variant="body2" color="text.secondary">
            Loading patient series...
          </Typography>
        </Stack>
      ) : null}

      {!loading && error ? <Alert severity="error">{error}</Alert> : null}

      {!loading && !error && selectedSeries ? (
        <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap alignItems="center">
          <Chip
            label={selectedSeries.phase ?? selectedSeries.type.toUpperCase()}
            color="primary"
            variant="outlined"
            size="small"
          />
          <Chip
            label={selectedSeries.has_seg ? 'SEG' : 'No SEG'}
            color={selectedSeries.has_seg ? 'secondary' : 'default'}
            variant={selectedSeries.has_seg ? 'filled' : 'outlined'}
            size="small"
          />
          <Chip
            label={selectedSeries.type === 'voi' ? 'VOI' : 'NIfTI'}
            color={selectedSeries.type === 'voi' ? 'success' : 'default'}
            variant={selectedSeries.type === 'voi' ? 'filled' : 'outlined'}
            size="small"
          />
          {selectedSeries.group ? (
            <Chip label={`Group ${selectedSeries.group}`} size="small" variant="outlined" />
          ) : null}
          {selectedSeries.laterality ? (
            <Chip
              label={`Side ${selectedSeries.laterality}`}
              size="small"
              variant="outlined"
            />
          ) : null}
        </Stack>
      ) : null}

      {!loading && !error && !selectedSeries ? (
        <Box>
          <Typography variant="body2" color="text.secondary">
            No series are available for this patient.
          </Typography>
        </Box>
      ) : null}
    </Stack>
  )
}

export default SeriesSelector
