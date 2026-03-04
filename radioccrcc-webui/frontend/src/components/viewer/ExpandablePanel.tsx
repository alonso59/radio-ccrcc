import { useEffect } from 'react'
import {
  Box,
  Button,
  Paper,
  Stack,
  Tooltip,
  Typography,
} from '@mui/material'

interface ExpandablePanelProps {
  axisLabel: string
  caption: string
  accent: string
  expanded: boolean
  onToggleExpand: () => void
  children: React.ReactNode
}

function ExpandablePanel({
  axisLabel,
  caption,
  accent,
  expanded,
  onToggleExpand,
  children,
}: ExpandablePanelProps) {
  useEffect(() => {
    if (!expanded) {
      return
    }

    function handleEscape(event: KeyboardEvent) {
      if (event.key === 'Escape') {
        onToggleExpand()
      }
    }

    window.addEventListener('keydown', handleEscape)
    return () => {
      window.removeEventListener('keydown', handleEscape)
    }
  }, [expanded, onToggleExpand])

  return (
    <Paper
      elevation={0}
      data-expanded={expanded ? 'true' : 'false'}
      data-panel={axisLabel}
      sx={{
        height: '100%',
        minHeight: 0,
        display: 'flex',
        flexDirection: 'column',
        overflow: 'hidden',
        borderColor: expanded ? accent : 'divider',
        boxShadow: expanded ? `0 0 0 1px ${accent}` : 'none',
        backgroundColor: 'background.paper',
      }}
    >
      <Stack
        direction="row"
        alignItems="center"
        justifyContent="space-between"
        spacing={1.5}
        onDoubleClick={onToggleExpand}
        sx={{
          px: 1.75,
          py: 1.25,
          borderBottom: '1px solid',
          borderColor: 'divider',
          backgroundColor: 'rgba(255, 255, 255, 0.02)',
          cursor: 'pointer',
          userSelect: 'none',
        }}
      >
        <Stack direction="row" spacing={1.25} alignItems="center">
          <Box
            sx={{
              px: 1,
              py: 0.35,
              borderRadius: 999,
              border: '1px solid',
              borderColor: accent,
              color: accent,
              backgroundColor: 'rgba(255,255,255,0.03)',
            }}
          >
            <Typography variant="caption" fontWeight={800}>
              {axisLabel}
            </Typography>
          </Box>
          <Typography variant="body2" color="text.secondary">
            {caption}
          </Typography>
        </Stack>

        <Tooltip title={expanded ? 'Restore grid' : 'Expand panel'}>
          <Button
            size="small"
            onClick={(event) => {
              event.stopPropagation()
              onToggleExpand()
            }}
            sx={{ color: 'text.secondary', minWidth: 0, px: 1.25 }}
          >
            {expanded ? 'Restore' : 'Expand'}
          </Button>
        </Tooltip>
      </Stack>

      <Box sx={{ flex: 1, minHeight: 0 }}>{children}</Box>
    </Paper>
  )
}

export default ExpandablePanel
