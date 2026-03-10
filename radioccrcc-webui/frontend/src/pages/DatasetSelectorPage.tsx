import { useEffect, useState } from 'react'
import {
  Alert,
  Box,
  Card,
  CardActionArea,
  CardContent,
  Chip,
  CircularProgress,
  Grid,
  Paper,
  Stack,
  Button,
  Typography,
} from '@mui/material'
import { useNavigate } from 'react-router-dom'

import { useSettings } from '../hooks/useSettings'
import {
  apiClient,
  type DatasetSummary,
  getApiErrorMessage,
} from '../services/api'

function DataBadge({
  label,
  active,
  tone,
}: {
  label: string
  active: boolean
  tone: 'primary' | 'secondary' | 'success'
}) {
  return (
    <Chip
      label={label}
      color={active ? tone : 'default'}
      size="small"
      variant={active ? 'filled' : 'outlined'}
      sx={{ fontWeight: 700, minWidth: 50 }}
    />
  )
}

function DatasetSelectorPage() {
  const navigate = useNavigate()
  const settingsState = useSettings()
  const [datasets, setDatasets] = useState<DatasetSummary[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let active = true

    apiClient
      .listDatasets()
      .then((response) => {
        if (!active) {
          return
        }
        setDatasets(response)
        setError(null)
      })
      .catch((requestError) => {
        if (!active) {
          return
        }
        setError(getApiErrorMessage(requestError))
      })
      .finally(() => {
        if (active) {
          setLoading(false)
        }
      })

    return () => {
      active = false
    }
  }, [])

  return (
    <Paper
      elevation={0}
      sx={{
        minHeight: 420,
        px: { xs: 3, md: 5 },
        py: { xs: 3, md: 4 },
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'space-between',
        background:
          'linear-gradient(160deg, rgba(125, 211, 252, 0.08), rgba(18, 18, 18, 0.98) 38%)',
      }}
    >
      <Stack spacing={3}>
        <Stack spacing={1.5} maxWidth={760}>
          <Typography variant="overline" color="text.secondary">
            Dataset Discovery
          </Typography>
          <Typography variant="h3">Dataset Selector</Typography>
          <Typography variant="body1" color="text.secondary">
            Available datasets are discovered live from the backend. Select a
            dataset to inspect its patient cohort and available radiology assets.
          </Typography>
        </Stack>

        {loading ? (
          <Stack
            direction="row"
            spacing={1.5}
            alignItems="center"
            sx={{ minHeight: 180 }}
          >
            <CircularProgress size={28} />
            <Typography color="text.secondary">Loading dataset inventory...</Typography>
          </Stack>
        ) : null}

        {!loading && error ? <Alert severity="error">{error}</Alert> : null}

        {!loading && !error ? (
          <Grid container spacing={2.5}>
            {datasets.map((dataset) => (
              <Grid size={{ xs: 12, md: 6 }} key={dataset.dataset_id}>
                <Card
                  sx={{
                    height: '100%',
                    background:
                      'linear-gradient(150deg, rgba(125, 211, 252, 0.08), rgba(18, 18, 18, 0.98) 42%)',
                  }}
                >
                  <CardActionArea
                    sx={{ height: '100%', alignItems: 'stretch' }}
                    onClick={() => navigate(`/datasets/${dataset.dataset_id}/patients`)}
                  >
                    <CardContent
                      sx={{
                        height: '100%',
                        p: 3,
                        display: 'flex',
                        flexDirection: 'column',
                        justifyContent: 'space-between',
                        gap: 3,
                      }}
                    >
                      <Stack spacing={1.5}>
                        <Stack
                          direction="row"
                          justifyContent="space-between"
                          alignItems="flex-start"
                          spacing={2}
                        >
                          <Box>
                            <Typography variant="overline" color="text.secondary">
                              Dataset
                            </Typography>
                            <Typography variant="h4">{dataset.dataset_id}</Typography>
                          </Box>
                          <Chip
                            label={`${dataset.patient_count} patients`}
                            color="primary"
                            variant="outlined"
                          />
                        </Stack>

                        <Typography
                          variant="body2"
                          color="text.secondary"
                          sx={{
                            display: '-webkit-box',
                            overflow: 'hidden',
                            WebkitLineClamp: 2,
                            WebkitBoxOrient: 'vertical',
                          }}
                        >
                          {dataset.has_manifest
                            ? 'Manifest detected for structured metadata import.'
                            : 'No manifest found; patient and series discovery is filename-driven.'}
                        </Typography>
                      </Stack>

                      <Stack spacing={1.5}>
                        <Typography variant="subtitle2" color="text.secondary">
                          Modalities and assets
                        </Typography>
                        <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
                          <DataBadge label="NIfTI" active={dataset.has_nifti} tone="primary" />
                          <DataBadge label="SEG" active={dataset.has_seg} tone="secondary" />
                          <DataBadge label="VOI" active={dataset.has_voi} tone="success" />
                          <Chip
                            label={dataset.has_manifest ? 'Manifest' : 'No manifest'}
                            size="small"
                            variant={dataset.has_manifest ? 'filled' : 'outlined'}
                          />
                        </Stack>
                        {!settingsState.loading &&
                        settingsState.allSettings[dataset.dataset_id]?.last_patient ? (
                          <Button
                            size="small"
                            variant="outlined"
                            onClick={(event) => {
                              event.preventDefault()
                              event.stopPropagation()
                              navigate(
                                `/datasets/${dataset.dataset_id}/patients/${settingsState.allSettings[dataset.dataset_id]?.last_patient}/viewer`,
                              )
                            }}
                          >
                            Resume{' '}
                            {settingsState.allSettings[dataset.dataset_id]?.last_patient}
                          </Button>
                        ) : null}
                      </Stack>
                    </CardContent>
                  </CardActionArea>
                </Card>
              </Grid>
            ))}
          </Grid>
        ) : null}
      </Stack>
    </Paper>
  )
}

export default DatasetSelectorPage
