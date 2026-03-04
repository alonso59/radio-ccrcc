import { createTheme } from '@mui/material/styles'

export const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#7dd3fc',
    },
    secondary: {
      main: '#fbbf24',
    },
    background: {
      default: '#1e1e1e',
      paper: '#121212',
    },
    divider: '#333333',
    text: {
      primary: '#dddddd',
      secondary: '#9ca3af',
    },
  },
  shape: {
    borderRadius: 18,
  },
  typography: {
    fontFamily: '"IBM Plex Sans", "Segoe UI", sans-serif',
    h3: {
      fontWeight: 700,
      letterSpacing: '-0.03em',
    },
    h4: {
      fontWeight: 700,
      letterSpacing: '-0.03em',
    },
    h5: {
      fontWeight: 650,
    },
    overline: {
      letterSpacing: '0.18em',
      fontWeight: 700,
    },
  },
  components: {
    MuiCssBaseline: {
      styleOverrides: {
        ':root': {
          colorScheme: 'dark',
        },
        'html, body, #root': {
          minHeight: '100%',
        },
        body: {
          margin: 0,
          backgroundColor: '#1e1e1e',
          backgroundImage:
            'radial-gradient(circle at top, rgba(125, 211, 252, 0.12), transparent 30%), linear-gradient(180deg, #191919 0%, #111111 100%)',
          color: '#dddddd',
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          border: '1px solid #333333',
          backgroundImage: 'none',
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 999,
          textTransform: 'none',
          fontWeight: 600,
        },
      },
    },
  },
})
