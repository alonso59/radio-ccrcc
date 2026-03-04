import { Button, Chip, Stack, Typography } from '@mui/material'

import { WINDOW_LEVEL_PRESETS } from './useWindowLevel'

interface WindowLevelControlProps {
  ww: number
  wl: number
  onPreset: (preset: keyof typeof WINDOW_LEVEL_PRESETS) => void
}

const PRESET_LABELS: Record<keyof typeof WINDOW_LEVEL_PRESETS, string> = {
  softTissue: 'Soft Tissue',
  bone: 'Bone',
  lung: 'Lung',
  brain: 'Brain',
}

function WindowLevelControl({ ww, wl, onPreset }: WindowLevelControlProps) {
  return (
    <Stack spacing={1.25} data-window-width={ww} data-window-level={wl}>
      <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap" useFlexGap>
        <Typography variant="overline" color="text.secondary">
          Window / Level
        </Typography>
        <Chip label={`WW ${ww}`} variant="outlined" size="small" />
        <Chip label={`WL ${wl}`} variant="outlined" size="small" />
        <Typography variant="caption" color="text.secondary">
          Right-drag any 2D panel to adjust contrast.
        </Typography>
      </Stack>

      <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
        {(Object.keys(WINDOW_LEVEL_PRESETS) as Array<keyof typeof WINDOW_LEVEL_PRESETS>).map(
          (preset) => (
            <Button
              key={preset}
              variant="outlined"
              size="small"
              onClick={() => onPreset(preset)}
            >
              {PRESET_LABELS[preset]}
            </Button>
          ),
        )}
      </Stack>
    </Stack>
  )
}

export default WindowLevelControl
