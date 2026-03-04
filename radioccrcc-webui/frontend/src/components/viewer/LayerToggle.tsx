import { Box, Checkbox, Stack, Typography } from '@mui/material'

interface LayerToggleProps {
  checked: boolean
  color: string
  disabled?: boolean
  label: string
  onChange: (checked: boolean) => void
}

function LayerToggle({
  checked,
  color,
  disabled = false,
  label,
  onChange,
}: LayerToggleProps) {
  return (
    <Stack
      direction="row"
      spacing={1}
      alignItems="center"
      data-layer-toggle={label.toLowerCase()}
    >
      <Checkbox
        checked={checked}
        disabled={disabled}
        onChange={(event) => onChange(event.target.checked)}
        size="small"
        sx={{ color, '&.Mui-checked': { color } }}
      />
      <Box
        sx={{
          width: 10,
          height: 10,
          borderRadius: '50%',
          backgroundColor: color,
          opacity: disabled ? 0.35 : 1,
        }}
      />
      <Typography
        variant="body2"
        color={disabled ? 'text.disabled' : 'text.primary'}
      >
        {label}
      </Typography>
    </Stack>
  )
}

export default LayerToggle
