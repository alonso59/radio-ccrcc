import {
  Box,
  Button,
  Chip,
  Divider,
  Paper,
  Stack,
  Typography,
} from '@mui/material'
import { useState } from 'react'
import { Link as RouterLink, useParams } from 'react-router-dom'

import type { SeriesInfo } from '../services/api'
import SeriesSelector from '../components/viewer/SeriesSelector'
import ViewerGrid2x2 from '../components/viewer/ViewerGrid2x2'

function ViewerPage() {
  const { dsid = 'unknown-dataset', pid = 'unknown-patient' } = useParams<{
    dsid: string
    pid: string
  }>()
  const [selectedSeries, setSelectedSeries] = useState<SeriesInfo | null>(null)

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
            <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
              {['Kidney', 'Tumor', 'Cyst'].map((label) => (
                <Chip key={label} label={label} variant="outlined" size="small" />
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
        </Stack>
      </Paper>

      <ViewerGrid2x2 />

      <Paper elevation={0} sx={{ px: { xs: 3, md: 4 }, py: 2.5 }}>
        <Stack
          direction={{ xs: 'column', lg: 'row' }}
          spacing={2}
          justifyContent="space-between"
          divider={<Divider orientation="vertical" flexItem sx={{ display: { xs: 'none', lg: 'block' } }} />}
        >
          <Box sx={{ flex: 1 }}>
            <Typography variant="overline" color="text.secondary">
              Bottom Bar
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Slice sliders, W/L presets, and opacity controls land here in M8.
            </Typography>
          </Box>
          <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
            {['Soft Tissue', 'Bone', 'Lung', 'Brain'].map((preset) => (
              <Chip key={preset} label={preset} variant="outlined" />
            ))}
          </Stack>
        </Stack>
      </Paper>
    </Stack>
  )
}

export default ViewerPage
