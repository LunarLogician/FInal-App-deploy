export const API_CONFIG = {
  MAIN_API_URL: (process.env.NEXT_PUBLIC_MAIN_API_URL || 'https://final-app-deploy-6d92.onrender.com').replace(/\/$/, ''),
  ENDPOINTS: {
    ANALYZE: process.env.NEXT_PUBLIC_ANALYZE_ENDPOINT || '/analyze',
    CONSISTENCY: process.env.NEXT_PUBLIC_CONSISTENCY_ENDPOINT || '/consistency/consistency',
    ESG: process.env.NEXT_PUBLIC_ESG_ENDPOINT || '/esg',
    UPLOAD: '/upload',
    CHAT: '/api/chat'
  }
};
