export const API_CONFIG = {
  MAIN_API_URL: (process.env.NEXT_PUBLIC_MAIN_API_URL || 'http://localhost:5001').replace(/\/$/, ''),
  SUMMARY_API_URL: (process.env.NEXT_PUBLIC_SUMMARY_API_URL || 'http://localhost:5004').replace(/\/$/, ''),
  ENDPOINTS: {
    ANALYZE: process.env.NEXT_PUBLIC_ANALYZE_ENDPOINT || '/analyze',
    CONSISTENCY: process.env.NEXT_PUBLIC_CONSISTENCY_ENDPOINT || '/consistency/consistency',
    ESG: process.env.NEXT_PUBLIC_ESG_ENDPOINT || '/esg',
    UPLOAD: '/upload',
    SUMMARIZE: '/summarize',
  }
};
