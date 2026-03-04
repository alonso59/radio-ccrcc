import { Slider, Stack, Typography } from '@mui/material'

interface OpacitySliderProps {
  color: string
  disabled?: boolean
  label: string
  onChange: (value: number) => void
  value: number
}

function OpacitySlider({
  color,
  disabled = false,
  label,
  onChange,
  value,
}: OpacitySliderProps) {
  return (
    <Stack spacing={0.75} sx={{ minWidth: 180 }}>
      <Stack direction="row" justifyContent="space-between" spacing={1}>
        <Typography variant="caption" color="text.secondary">
          {label}
        </Typography>
        <Typography variant="caption" sx={{ color }}>
          {value.toFixed(2)}
        </Typography>
      </Stack>
      <Slider
        size="small"
        disabled={disabled}
        min={0}
        max={1}
        step={0.01}
        value={value}
        onChange={(_event, nextValue) => onChange(nextValue as number)}
        sx={{ color }}
      />
    </Stack>
  )
}

export default OpacitySlider
