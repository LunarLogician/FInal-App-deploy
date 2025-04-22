import { useState, useEffect } from 'react';
import { API_CONFIG } from '../config/api';

interface AISummaryProps {
  text: string;
}

export default function AISummary({ text }: AISummaryProps) {
  const [summary, setSummary] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const generateSummary = async () => {
      if (!text) return;
      
      setIsLoading(true);
      setError(null);
      try {
        const response = await fetch(`${API_CONFIG.MAIN_API_URL}${API_CONFIG.ENDPOINTS.SUMMARIZE}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ 
            message: `Please summarize the following text into 3-5 key points: ${text}` 
          }),
        });

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        
        if (data.status === 'success' && data.response) {
          // Split the response into bullet points
          const points = data.response
            .split('\n')
            .map((point: string) => point.trim())
            .filter((point: string) => point.startsWith('-') || point.startsWith('•') || point.match(/^\d+\./))
            .map((point: string) => point.replace(/^[-•\d+\.\s]+/, '').trim());
          
          setSummary(points.length > 0 ? points : [data.response]);
        } else {
          throw new Error('Invalid response format');
        }
      } catch (error) {
        console.error('Error generating summary:', error);
        setError('Failed to generate summary. Please try again.');
        setSummary([]);
      } finally {
        setIsLoading(false);
      }
    };

    generateSummary();
  }, [text]);

  if (!text) return null;

  return (
    <div className="bg-white p-4 rounded-xl border border-zinc-200 mb-4">
      <h3 className="text-sm font-medium mb-3">📝 AI Summary</h3>
      {isLoading ? (
        <div className="text-sm text-zinc-500">Generating summary...</div>
      ) : error ? (
        <div className="text-sm text-red-500">{error}</div>
      ) : (
        <ul className="list-disc list-inside space-y-2 text-sm text-zinc-700">
          {summary.map((point, index) => (
            <li key={index}>{point}</li>
          ))}
        </ul>
      )}
    </div>
  );
} 