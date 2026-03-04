import { Box, Button, Chip, Paper, Stack, Typography } from '@mui/material'
import { Link as RouterLink, useParams } from 'react-router-dom'

function PatientListPage() {
  const { dsid = 'unknown-dataset' } = useParams<{ dsid: string }>()

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
      }}
    >
      <Stack spacing={2.5}>
        <Box>
          <Typography variant="overline" color="text.secondary">
            Dataset Context
          </Typography>
          <Stack direction="row" spacing={1.5} alignItems="center" flexWrap="wrap">
            <Typography variant="h3">Patient List</Typography>
            <Chip label={dsid} color="primary" variant="outlined" />
          </Stack>
        </Box>

        <Typography variant="body1" color="text.secondary" maxWidth={720}>
          This page is the dataset-scoped landing area for searchable patient
          tables in the next milestone. Route params are already active, and the
          page shell is consistent with the shared dark theme.
        </Typography>

        <Box
          sx={{
            p: 2.5,
            borderRadius: 4,
            border: '1px solid',
            borderColor: 'divider',
            backgroundColor: 'rgba(255, 255, 255, 0.02)',
          }}
        >
          <Typography variant="h6" sx={{ mb: 1 }}>
            Upcoming Work
          </Typography>
          <Typography variant="body2" color="text.secondary">
            M6 will replace this stub with a sortable patient table backed by
            real dataset discovery results.
          </Typography>
        </Box>
      </Stack>

      <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2} sx={{ mt: 4 }}>
        <Button component={RouterLink} to="/" variant="outlined">
          Back to Datasets
        </Button>
        <Button
          component={RouterLink}
          to={`/datasets/${dsid}/patients/case_00001/viewer`}
          variant="contained"
        >
          Open Viewer Stub
        </Button>
      </Stack>
    </Paper>
  )
}

export default PatientListPage
