export const API_CONFIG = {
  MAIN_API_URL: process.env.NEXT_PUBLIC_API_URL || 'https://final-app-deploy-6d92.onrender.com',
  ENDPOINTS: {
    ANALYZE: '/api/analyze',
    CONSISTENCY: '/api/consistency',
    ESG: '/api/esg',
    UPLOAD: '/api/upload',
    CHAT: '/api/chat',
    SUMMARIZE: '/api/chat'  // Using the chat endpoint for summarization
  }
};
