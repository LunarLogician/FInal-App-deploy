export const runtime = 'nodejs';

import { OpenAI } from 'openai';
import pinecone from '@/utils/pinecone';
import { RecordMetadata } from '@pinecone-database/pinecone';

import fs from 'fs';
import path from 'path';
import Papa from 'papaparse';

// ESG mapping type
type ESGScores = { [key: string]: number };

// Message type
type Message = {
  role: 'user' | 'assistant' | 'system';
  content: string;
  id?: string;
};

// Load and parse the mapping CSV at runtime
const mappingPath = path.join(process.cwd(), 'public', 'data', 'mapping-file-public.csv');
const mappingCsv = fs.readFileSync(mappingPath, 'utf8');
const { data: mapping } = Papa.parse(mappingCsv, {
  header: true,
  skipEmptyLines: true
});

const reportNameLookup = Object.fromEntries(
  mapping.map((row: any) => [row.pdf_name, row.report_name])
);

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

export async function POST(req: Request) {
  const {
    messages,
    docText,
    docName,
    consistencyResult,
    analysisResult,
    esgResult,
    tcfdAdvice,
    tcfdDrafts,
    csrdAdvice,
    csrdDrafts,
    griAdvice,
    griDrafts,
    sasbAdvice,
    sasbDrafts
  } = await req.json();

  // Declare these variables at the top so they're available everywhere:
  let sourceDocuments: string[] = [];
  let followUps: string[] = [];

  try {
    // Validate messages array
    if (!messages || !Array.isArray(messages) || messages.length === 0) {
      throw new Error('No valid messages provided');
    }

    // Get the last user message
    const lastUserMessage = messages[messages.length - 1];
    console.log('Processing chat request with message:', lastUserMessage.content.substring(0, 100) + '...');
    console.log('Uploaded doc:', docName);

    // If no document is provided, just use the messages array directly
    if (!docText || !docName) {
      const response = await openai.chat.completions.create({
        model: 'gpt-4-turbo-preview',
        messages: [
          { role: 'system', content: 'You are a helpful assistant. No document was uploaded. Respond helpfully and clearly.' },
          ...messages.map(msg => ({ role: msg.role, content: msg.content })),
        ],
        stream: true,
      });

      const stream = new ReadableStream({
        async start(controller) {
          const encoder = new TextEncoder();
          for await (const chunk of response) {
            const content = chunk.choices[0]?.delta?.content;
            if (content) {
              controller.enqueue(encoder.encode(content));
            }
          }
          controller.close();
        },
      });

      return new Response(stream, {
        headers: {
          'Content-Type': 'text/plain; charset=utf-8',
          'X-Source-Documents': JSON.stringify([]),
          'X-Extra-Suggestions': JSON.stringify([]),
        }
      });
    }

    // Get embeddings for the query
    const embedding = await openai.embeddings.create({
      model: "text-embedding-3-small",
      input: lastUserMessage.content,
      encoding_format: "float"
    });

    // Define namespace from document name
    const namespace = docName.replace(/[^a-zA-Z0-9-_]/g, '_').toLowerCase() || 'default';
    console.log('Querying Pinecone namespace:', namespace);
    
    const index = pinecone.index("embed-upload");
    const queryResponse = await index.namespace(namespace).query({
      vector: embedding.data[0].embedding,
      topK: 15,
      includeMetadata: true
    });

    console.log('Pinecone query results:', queryResponse.matches.length);

    let context = queryResponse.matches
      .filter(match => match.metadata?.text)
      .map(match => {
        const page = match.metadata?.page_number ? ` (p. ${match.metadata.page_number})` : '';
        return `${match.metadata?.text || ''}`;
      })
      .join('\n\n');

    if (!context) {
      // If no context from Pinecone, use the docText directly
      context = docText;
      console.log('Using direct document text as context');
    }

    // Build a mapping from doc name to a Set of pages
    const sourceMap: { [doc: string]: Set<number> } = {};
    for (const match of queryResponse.matches) {
      const doc = match.id.split('/')[0] || namespace;
      const page = match.metadata?.page_number;
      if (typeof page === 'number') {
        if (!sourceMap[doc]) sourceMap[doc] = new Set();
        sourceMap[doc].add(page);
      }
    }

    // Format the sources nicely
    sourceDocuments = Object.entries(sourceMap).map(([doc, pages]) => {
      const sortedPages = Array.from(pages).sort((a, b) => a - b);
      const pageList = sortedPages.join(', ');
      return `${doc}, pp. ${pageList}`;
    });

    const friendlyName = reportNameLookup[docName] || docName;

    // Create the system message with context
    const systemMessage = {
      role: 'system' as const,
      content: `You are an AI assistant analyzing a document. The document is titled "${docName}".

      When answering questions:
      1. Only use information that is explicitly stated in the document
      2. If something isn't mentioned in the document, say "The report does not mention..."
      3. When citing information, include page numbers if available (e.g. "On page 3...")
      4. Be specific and accurate - don't make assumptions beyond what's in the text
      5. Use direct quotes when appropriate

      Here is the relevant context from the document:

      ${context}
      
      ${analysisResult || consistencyResult || esgResult ? `\nAnalysis Summary:\n${JSON.stringify({ analysisResult, consistencyResult, esgResult }, null, 2)}` : ''}`
    };

    // Create the chat completion with streaming
    const response = await openai.chat.completions.create({
      model: 'gpt-4-turbo-preview',
      messages: [
        systemMessage,
        ...messages.map(msg => ({ role: msg.role, content: msg.content }))
      ],
      stream: true,
    });

    const stream = new ReadableStream({
      async start(controller) {
        const encoder = new TextEncoder();
        for await (const chunk of response) {
          const content = chunk.choices[0]?.delta?.content;
          if (content) {
            controller.enqueue(encoder.encode(content));
          }
        }
        controller.close();
      },
    });

    return new Response(stream, {
      headers: {
        'Content-Type': 'text/plain; charset=utf-8',
        'X-Source-Documents': JSON.stringify(sourceDocuments),
        'X-Extra-Suggestions': JSON.stringify(followUps),
      }
    });

  } catch (error: any) {
    console.error('Error in chat endpoint:', error);
    return new Response(JSON.stringify({ error: error.message }), {
      status: 500,
      headers: { 'Content-Type': 'application/json' }
    });
  }
}
