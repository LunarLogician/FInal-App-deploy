export const API_CONFIG = {
  MAIN_API_URL: process.env.NEXT_PUBLIC_API_URL || 'https://final-app-deploy-6d92.onrender.com',
  ENDPOINTS: {
    ANALYZE: '/analyze',
    CONSISTENCY: '/consistency',
    ESG: '/esg',
    UPLOAD: '/upload',
    CHAT: '/api/chat',
    SUMMARIZE: '/api/chat'  // Using the chat endpoint for summarization
  }
};
