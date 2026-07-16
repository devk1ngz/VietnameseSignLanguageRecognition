import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'

// Validate required environment variables at runtime
const requiredEnvVars = ['VITE_WS_URL', 'VITE_API_URL']
requiredEnvVars.forEach(key => {
  if (!import.meta.env[key]) {
    console.error(`[Configuration Error] Missing required env var: ${key}`)
    // We throw an error to halt execution if they are missing
    throw new Error(`Missing required environment variable: ${key}`)
  }
})


createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
