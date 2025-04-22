import { OpenAI } from 'openai';
import { NextResponse } from 'next/server';
import pinecone from '@/utils/pinecone';

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// Function to chunk text into smaller pieces
function chunkText(text: string, maxChunkSize: number = 1000): string[] {
  const sentences = text.match(/[^.!?]+[.!?]+/g) || [];
  const chunks: string[] = [];
  let currentChunk = '';

  for (const sentence of sentences) {
    if ((currentChunk + sentence).length <= maxChunkSize) {
      currentChunk += sentence;
    } else {
      if (currentChunk) chunks.push(currentChunk.trim());
      currentChunk = sentence;
    }
  }
  if (currentChunk) chunks.push(currentChunk.trim());
  return chunks;
}

export async function POST(req: Request) {
  try {
    const { text, namespace } = await req.json();
    
    if (!text || !namespace) {
      return NextResponse.json(
        { error: 'Text and namespace are required' },
        { status: 400 }
      );
    }

    // Split text into chunks
    const chunks = chunkText(text);
    
    // Get embeddings for all chunks
    const embeddings = await Promise.all(
      chunks.map(async (chunk, index) => {
        const embedding = await openai.embeddings.create({
          model: "text-embedding-3-small",
          input: chunk,
          encoding_format: "float"
        });
        return {
          id: `${namespace}/${index}`,
          values: embedding.data[0].embedding,
          metadata: {
            text: chunk,
            page_number: Math.floor(index / 3) + 1 // Rough page estimation
          }
        };
      })
    );

    // Index embeddings in Pinecone
    const index = pinecone.index("embed-upload");
    await index.namespace(namespace).upsert(embeddings);

    return NextResponse.json({ success: true, chunks: chunks.length });
  } catch (error: any) {
    console.error('Error in Pinecone upload:', error);
    return NextResponse.json(
      { error: error.message },
      { status: 500 }
    );
  }
} 