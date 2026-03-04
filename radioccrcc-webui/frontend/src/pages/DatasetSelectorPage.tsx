import { Box, Button, Paper, Stack, Typography } from '@mui/material'
import { Link as RouterLink } from 'react-router-dom'

function DatasetSelectorPage() {
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
      <Stack spacing={2.5} maxWidth={720}>
        <Typography variant="overline" color="text.secondary">
          Frontend Shell
        </Typography>
        <Typography variant="h3">Dataset Selector</Typography>
        <Typography variant="body1" color="text.secondary">
          This stub page anchors the first navigation step of the viewer. It is
          wired for routing, dark theme rendering, and the shared API layer.
        </Typography>
      </Stack>

      <Box
        sx={{
          mt: 4,
          p: 2.5,
          borderRadius: 4,
          border: '1px solid',
          borderColor: 'divider',
          backgroundColor: 'rgba(255, 255, 255, 0.02)',
        }}
      >
        <Stack
          direction={{ xs: 'column', sm: 'row' }}
          spacing={2}
          justifyContent="space-between"
          alignItems={{ xs: 'flex-start', sm: 'center' }}
        >
          <Box>
            <Typography variant="h6">Route Preview</Typography>
            <Typography variant="body2" color="text.secondary">
              Continue to the patient list page for a dataset-scoped workflow.
            </Typography>
          </Box>
          <Button
            component={RouterLink}
            to="/datasets/Dataset820/patients"
            variant="contained"
            color="primary"
          >
            Open Patient List
          </Button>
        </Stack>
      </Box>
    </Paper>
  )
}

export default DatasetSelectorPage
