import { Slider, Stack, Typography } from '@mui/material'

interface BlendSliderProps {
  disabled?: boolean
  onChange: (value: number) => void
  value: number
}

function BlendSlider({
  disabled = false,
  onChange,
  value,
}: BlendSliderProps) {
  return (
    <Stack spacing={0.75} sx={{ minWidth: 200 }} data-blend-slider="root">
      <Stack direction="row" justifyContent="space-between" spacing={1}>
        <Typography variant="caption" color="text.secondary">
          3D Blend
        </Typography>
        <Typography variant="caption" color="text.secondary">
          {value.toFixed(2)}
        </Typography>
      </Stack>
      <Slider
        data-blend-slider="control"
        size="small"
        min={0}
        max={1}
        step={0.01}
        value={value}
        disabled={disabled}
        onChange={(_event, nextValue) => onChange(nextValue as number)}
      />
    </Stack>
  )
}

export default BlendSlider
