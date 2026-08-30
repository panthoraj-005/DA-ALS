import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The backend runs separately (uvicorn on :8000). In dev we proxy so the
// browser talks to one origin and CORS never gets in the way; VITE_API_BASE
// overrides this for a deployed build.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: process.env.VITE_API_TARGET ?? 'http://127.0.0.1:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
})
