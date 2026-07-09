import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // Ohne dies bindet Vite je nach Node-DNS-Reihenfolge nur an ::1 (IPv6) statt
    // auch an 127.0.0.1 (IPv4) — FRONTEND_URL in .env zeigt aber auf 127.0.0.1,
    // wodurch z.B. E-Mail-Verifizierungslinks "Verbindung abgelehnt" ergeben.
    host: true,
  },
})
