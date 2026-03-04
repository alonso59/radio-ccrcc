import { useEffect, useState } from 'react'
import {
  AppBar,
  Box,
  Button,
  Chip,
  Container,
  Stack,
  Toolbar,
  Typography,
} from '@mui/material'
import { Link as RouterLink, Route, Routes, useLocation } from 'react-router-dom'

import DatasetSelectorPage from './pages/DatasetSelectorPage'
import PatientListPage from './pages/PatientListPage'
import ViewerPage from './pages/ViewerPage'
import { apiClient } from './services/api'

type ApiStatus = 'checking' | 'online' | 'offline'

function App() {
  const location = useLocation()
  const [apiStatus, setApiStatus] = useState<ApiStatus>('checking')

  useEffect(() => {
    let active = true

    apiClient
      .getHealth()
      .then(() => {
        if (active) {
          setApiStatus('online')
        }
      })
      .catch(() => {
        if (active) {
          setApiStatus('offline')
        }
      })

    return () => {
      active = false
    }
  }, [location.pathname])

  return (
    <Box sx={{ minHeight: '100vh' }}>
      <AppBar
        position="sticky"
        elevation={0}
        sx={{
          borderBottom: '1px solid',
          borderColor: 'divider',
          backgroundColor: 'rgba(18, 18, 18, 0.82)',
          backdropFilter: 'blur(18px)',
        }}
      >
        <Toolbar sx={{ gap: 2, flexWrap: 'wrap', py: 1 }}>
          <Box sx={{ flexGrow: 1 }}>
            <Typography variant="overline" color="text.secondary">
              Radiology WebUI
            </Typography>
            <Typography variant="h6">Routing Shell</Typography>
          </Box>

          <Stack direction="row" spacing={1.25} flexWrap="wrap" useFlexGap>
            <Button component={RouterLink} to="/" color="inherit">
              Datasets
            </Button>
            <Button
              component={RouterLink}
              to="/datasets/Dataset820/patients"
              color="inherit"
            >
              Patients
            </Button>
            <Button
              component={RouterLink}
              to="/datasets/Dataset820/patients/case_00001/viewer"
              color="inherit"
            >
              Viewer
            </Button>
          </Stack>

          <Chip
            color={
              apiStatus === 'online'
                ? 'primary'
                : apiStatus === 'offline'
                  ? 'secondary'
                  : 'default'
            }
            label={
              apiStatus === 'online'
                ? 'API online'
                : apiStatus === 'offline'
                  ? 'API unreachable'
                  : 'Checking API'
            }
            variant={apiStatus === 'checking' ? 'outlined' : 'filled'}
          />
        </Toolbar>
      </AppBar>

      <Container maxWidth="lg" sx={{ py: { xs: 3, md: 5 } }}>
        <Stack spacing={2.5}>
          <Box>
            <Typography variant="overline" color="text.secondary">
              Active Route
            </Typography>
            <Typography variant="body1">{location.pathname}</Typography>
          </Box>

          <Routes>
            <Route path="/" element={<DatasetSelectorPage />} />
            <Route path="/datasets/:dsid/patients" element={<PatientListPage />} />
            <Route
              path="/datasets/:dsid/patients/:pid/viewer"
              element={<ViewerPage />}
            />
          </Routes>
        </Stack>
      </Container>
    </Box>
  )
}

export default App
