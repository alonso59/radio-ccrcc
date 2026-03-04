import { Box, Button, Grid, Paper, Stack, Typography } from '@mui/material'
import { Link as RouterLink, useParams } from 'react-router-dom'

function ViewerPage() {
  const { dsid = 'unknown-dataset', pid = 'unknown-patient' } = useParams<{
    dsid: string
    pid: string
  }>()

  return (
    <Stack spacing={3}>
      <Paper
        elevation={0}
        sx={{
          px: { xs: 3, md: 5 },
          py: { xs: 3, md: 4 },
        }}
      >
        <Typography variant="overline" color="text.secondary">
          Viewer Route
        </Typography>
        <Typography variant="h3" sx={{ mt: 1 }}>
          Viewer Page
        </Typography>
        <Typography variant="body1" color="text.secondary" sx={{ mt: 2, maxWidth: 760 }}>
          The 2×2 diagnostic workspace lands here in later milestones. The route
          already carries the selected dataset and patient context.
        </Typography>

        <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1.5} sx={{ mt: 3 }}>
          <Button component={RouterLink} to={`/datasets/${dsid}/patients`} variant="outlined">
            Back to Patient List
          </Button>
          <Button component={RouterLink} to="/" variant="contained">
            Return Home
          </Button>
        </Stack>
      </Paper>

      <Grid container spacing={2}>
        {[
          ['Dataset', dsid],
          ['Patient', pid],
          ['Viewer Mode', 'Stub shell'],
          ['Next Milestone', 'M6 dataset and patient pages'],
        ].map(([label, value]) => (
          <Grid size={{ xs: 12, sm: 6 }} key={label}>
            <Paper elevation={0} sx={{ p: 2.5, minHeight: 132 }}>
              <Typography variant="overline" color="text.secondary">
                {label}
              </Typography>
              <Box sx={{ mt: 1.5 }}>
                <Typography variant="h5">{value}</Typography>
              </Box>
            </Paper>
          </Grid>
        ))}
      </Grid>
    </Stack>
  )
}

export default ViewerPage
