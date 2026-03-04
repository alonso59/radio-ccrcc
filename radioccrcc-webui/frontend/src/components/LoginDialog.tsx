import { useState } from 'react'
import {
  Button,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  Stack,
  TextField,
  Typography,
} from '@mui/material'

interface LoginDialogProps {
  open: boolean
  initialToken: string
  onCancel: () => void
  onSubmit: (token: string) => void
}

function LoginDialog({
  open,
  initialToken,
  onCancel,
  onSubmit,
}: LoginDialogProps) {
  const [token, setToken] = useState(initialToken)

  return (
    <Dialog open={open} onClose={onCancel} fullWidth maxWidth="xs">
      <DialogTitle>Authentication Required</DialogTitle>
      <DialogContent>
        <Stack spacing={2} sx={{ pt: 1 }}>
          <Typography variant="body2" color="text.secondary">
            This API is protected by `RADIOLOGY_UI_TOKEN`. Enter the shared
            bearer token to continue.
          </Typography>
          <TextField
            autoFocus
            fullWidth
            label="Bearer token"
            type="password"
            value={token}
            onChange={(event) => setToken(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter' && token.trim()) {
                onSubmit(token.trim())
              }
            }}
          />
        </Stack>
      </DialogContent>
      <DialogActions sx={{ px: 3, pb: 2.5 }}>
        <Button onClick={onCancel} color="inherit">
          Cancel
        </Button>
        <Button
          onClick={() => onSubmit(token.trim())}
          variant="contained"
          disabled={!token.trim()}
        >
          Retry Request
        </Button>
      </DialogActions>
    </Dialog>
  )
}

export default LoginDialog
