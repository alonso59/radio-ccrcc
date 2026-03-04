import { Box, Stack, Typography } from '@mui/material'
import { useState, type ReactNode } from 'react'

import ExpandablePanel from './ExpandablePanel'

const PANEL_DEFS = [
  {
    id: 'axial',
    label: 'AXIAL',
    accent: '#fbbf24',
    caption: 'Axial viewport placeholder',
    placeholder: 'Slice view and navigation land here in M8.',
  },
  {
    id: 'sagittal',
    label: 'SAGITTAL',
    accent: '#22c55e',
    caption: 'Sagittal viewport placeholder',
    placeholder: 'Crosshair sync and scrolling arrive next.',
  },
  {
    id: 'coronal',
    label: 'CORONAL',
    accent: '#ef4444',
    caption: 'Coronal viewport placeholder',
    placeholder: 'Coronal image panel will reuse the shared slice API.',
  },
  {
    id: 'surface',
    label: '3D SURFACE',
    accent: '#f5f5f5',
    caption: 'Surface viewport placeholder',
    placeholder: 'Mesh rendering and blend controls arrive in M9.',
  },
] as const

type PanelId = (typeof PANEL_DEFS)[number]['id']

interface PanelOverride {
  caption?: string
  content?: ReactNode
}

interface ViewerGrid2x2Props {
  panels?: Partial<Record<PanelId, PanelOverride>>
}

function ViewerGrid2x2({ panels = {} }: ViewerGrid2x2Props) {
  const [expandedPanel, setExpandedPanel] = useState<PanelId | null>(null)

  return (
    <Box
      sx={{
        display: 'grid',
        gridTemplateColumns: { xs: '1fr', lg: '1fr 1fr' },
        gridTemplateRows: { xs: 'repeat(4, minmax(240px, 1fr))', lg: 'repeat(2, minmax(280px, 1fr))' },
        gap: '2px',
        p: '2px',
        borderRadius: 4,
        backgroundColor: 'divider',
        minHeight: { xs: 980, lg: 760 },
      }}
    >
      {PANEL_DEFS.map((panel) => {
        const expanded = expandedPanel === panel.id
        const hidden = expandedPanel !== null && !expanded

        return (
          <Box
            key={panel.id}
            sx={{
              minHeight: 0,
              display: hidden ? 'none' : 'block',
              ...(expandedPanel
                ? {
                    gridColumn: '1 / -1',
                    gridRow: '1 / -1',
                  }
                : {}),
            }}
          >
            <ExpandablePanel
              axisLabel={panel.label}
              caption={panels[panel.id]?.caption ?? panel.caption}
              accent={panel.accent}
              expanded={expanded}
              onToggleExpand={() =>
                setExpandedPanel((current) => (current === panel.id ? null : panel.id))
              }
            >
              {panels[panel.id]?.content ?? (
                <Stack
                  spacing={2}
                  justifyContent="space-between"
                  sx={{
                    height: '100%',
                    p: 2.5,
                    background:
                      'linear-gradient(180deg, rgba(255,255,255,0.015), rgba(255,255,255,0.03))',
                  }}
                >
                  <Box
                    sx={{
                      flex: 1,
                      minHeight: 0,
                      borderRadius: 3,
                      border: '1px dashed',
                      borderColor: 'divider',
                      background:
                        'radial-gradient(circle at top, rgba(125, 211, 252, 0.08), transparent 42%), rgba(255,255,255,0.015)',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      textAlign: 'center',
                      px: 3,
                    }}
                  >
                    <Stack spacing={1.25} alignItems="center">
                      <Typography variant="h5">{panel.label}</Typography>
                      <Typography variant="body2" color="text.secondary">
                        {panel.placeholder}
                      </Typography>
                    </Stack>
                  </Box>

                  <Stack direction="row" justifyContent="space-between" alignItems="center">
                    <Typography variant="caption" color="text.secondary">
                      Double-click header or use the corner control to expand.
                    </Typography>
                    <Typography
                      variant="caption"
                      sx={{
                        px: 1,
                        py: 0.4,
                        borderRadius: 999,
                        border: '1px solid',
                        borderColor: panel.accent,
                        color: panel.accent,
                      }}
                    >
                      Placeholder
                    </Typography>
                  </Stack>
                </Stack>
              )}
            </ExpandablePanel>
          </Box>
        )
      })}
    </Box>
  )
}

export default ViewerGrid2x2
