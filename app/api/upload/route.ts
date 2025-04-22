import { NextRequest, NextResponse } from 'next/server';
import { writeFile, unlink } from 'fs/promises';
import { spawn } from 'child_process';
import path from 'path';

export async function POST(req: NextRequest) {
  const data = await req.formData();
  const file = data.get('file') as File;

  // Convert the uploaded file to a buffer
  const bytes = await file.arrayBuffer();

  // Validate allowed file extensions
  const ext = file.name.slice(file.name.lastIndexOf(".")).toLowerCase();
  const allowed = [".pdf", ".docx"];
  if (!allowed.includes(ext)) {
    return NextResponse.json(
      { error: "Unsupported file type. Please upload a PDF or Word document (.pdf, .docx)" },
      { status: 400 }
    );
  }
  
  const buffer = Buffer.from(bytes);

  // Write to a temporary location
  const tempPath = `/tmp/${file.name}`;
  await writeFile(tempPath, buffer);

  try {
    // Determine the path to the Python script in the scripts folder
    const pyScript = path.join(process.cwd(), 'scripts', 'extract_text.py');

    // Spawn a child process to run the Python script with the temp file path as argument
    const proc = spawn('python3', [pyScript, tempPath]);

    let output = '';
    proc.stdout.on('data', (data) => {
      output += data.toString();
    });

    proc.stderr.on('data', (err) => {
      console.error('Python error:', err.toString());
    });

    // Wait until the Python script completes
    await new Promise((resolve) => {
      proc.on('close', resolve);
    });

    // Return a JSON response with the file name and extracted text
    return NextResponse.json({
      name: file.name,
      text: output,
    });
  } finally {
    // Clean up the temporary file
    await unlink(tempPath);
  }
}
