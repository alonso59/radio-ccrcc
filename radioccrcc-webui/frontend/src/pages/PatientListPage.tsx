import { useDeferredValue, useEffect, useState } from 'react'
import {
  Alert,
  Box,
  Button,
  Chip,
  CircularProgress,
  FormControl,
  InputLabel,
  MenuItem,
  Paper,
  Select,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TableSortLabel,
  TextField,
  Typography,
} from '@mui/material'
import { Link as RouterLink, useNavigate, useParams } from 'react-router-dom'

import {
  apiClient,
  type PatientSummary,
  getApiErrorMessage,
} from '../services/api'

type SortField =
  | 'patient_id'
  | 'group'
  | 'series_count'
  | 'seg_count'
  | 'voi_count'

type SortDirection = 'asc' | 'desc'

function PatientListPage() {
  const { dsid = 'unknown-dataset' } = useParams<{ dsid: string }>()
  const navigate = useNavigate()
  const [requestState, setRequestState] = useState<{
    datasetId: string | null
    patients: PatientSummary[]
    error: string | null
  }>({
    datasetId: null,
    patients: [],
    error: null,
  })
  const [search, setSearch] = useState('')
  const [groupFilter, setGroupFilter] = useState('all')
  const [phaseFilter, setPhaseFilter] = useState('all')
  const [sortField, setSortField] = useState<SortField>('patient_id')
  const [sortDirection, setSortDirection] = useState<SortDirection>('asc')
  const deferredSearch = useDeferredValue(search)
  const loading = requestState.datasetId !== dsid
  const patients = requestState.datasetId === dsid ? requestState.patients : []
  const error = requestState.datasetId === dsid ? requestState.error : null

  useEffect(() => {
    let active = true

    apiClient
      .listPatients(dsid)
      .then((response) => {
        if (!active) {
          return
        }
        setRequestState({
          datasetId: dsid,
          patients: response,
          error: null,
        })
      })
      .catch((requestError) => {
        if (!active) {
          return
        }
        setRequestState({
          datasetId: dsid,
          patients: [],
          error: getApiErrorMessage(requestError),
        })
      })

    return () => {
      active = false
    }
  }, [dsid])

  const groupOptions = Array.from(
    new Set(
      patients.map((patient) => patient.group ?? 'Unknown'),
    ),
  ).sort((left, right) => left.localeCompare(right, undefined, { sensitivity: 'base' }))

  const phaseOptions = Array.from(
    new Set(patients.flatMap((patient) => patient.phases)),
  ).sort((left, right) => left.localeCompare(right, undefined, { sensitivity: 'base' }))

  const normalizedSearch = deferredSearch.trim().toLowerCase()
  const filteredPatients = patients
    .filter((patient) => {
      if (normalizedSearch && !patient.patient_id.toLowerCase().includes(normalizedSearch)) {
        return false
      }
      if (groupFilter !== 'all' && (patient.group ?? 'Unknown') !== groupFilter) {
        return false
      }
      if (phaseFilter !== 'all' && !patient.phases.includes(phaseFilter)) {
        return false
      }
      return true
    })
    .sort((left, right) => comparePatients(left, right, sortField, sortDirection))

  function toggleSort(field: SortField) {
    if (field === sortField) {
      setSortDirection((current) => (current === 'asc' ? 'desc' : 'asc'))
      return
    }
    setSortField(field)
    setSortDirection('asc')
  }

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

        <Typography variant="body1" color="text.secondary" maxWidth={840}>
          Filter and sort the discovered patient cohort for <strong>{dsid}</strong>,
          then open a patient directly in the viewer workflow.
        </Typography>

        <Stack direction={{ xs: 'column', md: 'row' }} spacing={2}>
          <TextField
            label="Search patient ID"
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            fullWidth
          />
          <FormControl sx={{ minWidth: { xs: '100%', md: 200 } }}>
            <InputLabel id="group-filter-label">Group</InputLabel>
            <Select
              labelId="group-filter-label"
              label="Group"
              value={groupFilter}
              onChange={(event) => setGroupFilter(event.target.value)}
            >
              <MenuItem value="all">All groups</MenuItem>
              {groupOptions.map((group) => (
                <MenuItem key={group} value={group}>
                  {group}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
          <FormControl sx={{ minWidth: { xs: '100%', md: 220 } }}>
            <InputLabel id="phase-filter-label">Phase</InputLabel>
            <Select
              labelId="phase-filter-label"
              label="Phase"
              value={phaseFilter}
              onChange={(event) => setPhaseFilter(event.target.value)}
            >
              <MenuItem value="all">All phases</MenuItem>
              {phaseOptions.map((phase) => (
                <MenuItem key={phase} value={phase}>
                  {phase}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Stack>

        {loading ? (
          <Stack
            direction="row"
            spacing={1.5}
            alignItems="center"
            sx={{ minHeight: 180 }}
          >
            <CircularProgress size={28} />
            <Typography color="text.secondary">Loading patient table...</Typography>
          </Stack>
        ) : null}

        {!loading && error ? <Alert severity="error">{error}</Alert> : null}

        {!loading && !error ? (
          <TableContainer
            sx={{
              border: '1px solid',
              borderColor: 'divider',
              borderRadius: 4,
              backgroundColor: 'rgba(255, 255, 255, 0.02)',
            }}
          >
            <Table>
              <TableHead>
                <TableRow>
                  <SortableHeader
                    active={sortField === 'patient_id'}
                    direction={sortDirection}
                    onClick={() => toggleSort('patient_id')}
                  >
                    Patient ID
                  </SortableHeader>
                  <SortableHeader
                    active={sortField === 'group'}
                    direction={sortDirection}
                    onClick={() => toggleSort('group')}
                  >
                    Group
                  </SortableHeader>
                  <TableCell>Phases</TableCell>
                  <SortableHeader
                    align="right"
                    active={sortField === 'series_count'}
                    direction={sortDirection}
                    onClick={() => toggleSort('series_count')}
                  >
                    Series
                  </SortableHeader>
                  <SortableHeader
                    align="right"
                    active={sortField === 'seg_count'}
                    direction={sortDirection}
                    onClick={() => toggleSort('seg_count')}
                  >
                    Seg
                  </SortableHeader>
                  <SortableHeader
                    align="right"
                    active={sortField === 'voi_count'}
                    direction={sortDirection}
                    onClick={() => toggleSort('voi_count')}
                  >
                    VOI
                  </SortableHeader>
                </TableRow>
              </TableHead>
              <TableBody>
                {filteredPatients.map((patient) => (
                  <TableRow
                    hover
                    key={patient.patient_id}
                    onClick={() =>
                      navigate(`/datasets/${dsid}/patients/${patient.patient_id}/viewer`)
                    }
                    sx={{
                      cursor: 'pointer',
                      '&:last-child td': {
                        borderBottom: 0,
                      },
                    }}
                  >
                    <TableCell>
                      <Stack spacing={0.5}>
                        <Typography fontWeight={600}>{patient.patient_id}</Typography>
                        {patient.source_patient_id ? (
                          <Typography variant="caption" color="text.secondary">
                            Source: {patient.source_patient_id}
                          </Typography>
                        ) : null}
                      </Stack>
                    </TableCell>
                    <TableCell>{patient.group ?? 'Unknown'}</TableCell>
                    <TableCell>
                      <Stack direction="row" spacing={0.75} flexWrap="wrap" useFlexGap>
                        {patient.phases.length > 0 ? (
                          patient.phases.map((phase) => (
                            <Chip key={phase} label={phase} size="small" variant="outlined" />
                          ))
                        ) : (
                          <Typography variant="body2" color="text.secondary">
                            None
                          </Typography>
                        )}
                      </Stack>
                    </TableCell>
                    <TableCell align="right">{patient.series_count}</TableCell>
                    <TableCell align="right">{patient.seg_count}</TableCell>
                    <TableCell align="right">{patient.voi_count}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        ) : null}

        {!loading && !error && filteredPatients.length === 0 ? (
          <Alert severity="info">
            No patients match the current search and filter combination.
          </Alert>
        ) : null}
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

function SortableHeader({
  active,
  align,
  children,
  direction,
  onClick,
}: {
  active: boolean
  align?: 'left' | 'right'
  children: string
  direction: SortDirection
  onClick: () => void
}) {
  return (
    <TableCell align={align}>
      <TableSortLabel
        active={active}
        direction={active ? direction : 'asc'}
        onClick={onClick}
      >
        {children}
      </TableSortLabel>
    </TableCell>
  )
}

function comparePatients(
  left: PatientSummary,
  right: PatientSummary,
  field: SortField,
  direction: SortDirection,
) {
  const multiplier = direction === 'asc' ? 1 : -1

  if (field === 'group') {
    const leftValue = left.group ?? 'Unknown'
    const rightValue = right.group ?? 'Unknown'
    return (
      leftValue.localeCompare(rightValue, undefined, {
        sensitivity: 'base',
        numeric: true,
      }) * multiplier
    )
  }

  if (field === 'patient_id') {
    return (
      left.patient_id.localeCompare(right.patient_id, undefined, {
        sensitivity: 'base',
        numeric: true,
      }) * multiplier
    )
  }

  return (left[field] - right[field]) * multiplier
}

export default PatientListPage
