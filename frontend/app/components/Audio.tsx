import React, { useState, useEffect } from 'react';

const Audio: React.FC = () => {
  const [audioContext, setAudioContext] = useState<AudioContext | null>(null);
  const [audioInput, setAudioInput] = useState<MediaStream | null>(null);
  const [socket, setSocket] = useState<WebSocket | null>(null);

  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8765');
    ws.binaryType = 'arraybuffer';
    ws.onopen = () => {
      console.log('WebSocket connection established');
    };
    ws.onclose = (event) => {
      console.log('WebSocket connection closed', event);
    };
    ws.onerror = (error) => {
      console.error('WebSocket error', error);
    };
    ws.onmessage = (event) => {
      console.log('Received buffer:', event.data);
      playAudio(event.data);
    };
    setSocket(ws);

    return () => {
      ws.close();
    };
  }, []);

  const startRecording = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const AudioContext = window.AudioContext || (window as any).webkitAudioContext;
    const context = new AudioContext();
    const source = context.createMediaStreamSource(stream);
    const processor = context.createScriptProcessor(4096, 1, 1);

    processor.onaudioprocess = (e) => {
      if (socket && socket.readyState === WebSocket.OPEN) {
        const inputData = e.inputBuffer.getChannelData(0);
        const int16Data = new Int16Array(inputData.length);
        for (let i = 0; i < inputData.length; i++) {
          int16Data[i] = inputData[i] * 0x7FFF;
        }
        socket.send(int16Data.buffer);
      }
    };

    source.connect(processor);
    processor.connect(context.destination);

    setAudioContext(context);
    setAudioInput(stream);
  };

  const stopRecording = () => {
    if (audioInput) {
      audioInput.getTracks().forEach((track: MediaStreamTrack) => track.stop());
    }
    if (audioContext) {
      audioContext.close();
    }
  };

  const playAudio = (buffer: ArrayBuffer) => {
    if (buffer.byteLength === 0) {
      console.error('Received empty buffer');
      return;
    }
  
    const AudioContext = window.AudioContext || (window as any).webkitAudioContext;
    let context = audioContext;
  
    if (!context || context.state === 'closed') {
      context = new AudioContext();
      setAudioContext(context);
    }
  
    const audioData = new Float32Array(buffer.byteLength / 2);
    const view = new DataView(buffer);
  
    for (let i = 0; i < audioData.length; i++) {
      audioData[i] = view.getInt16(i * 2, true) / 0x7FFF;
    }
  
    const audioBuffer = context.createBuffer(1, audioData.length, 44100);
    audioBuffer.copyToChannel(audioData, 0);
  
    const source = context.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(context.destination);
    source.start(0);
  };

  return (
    <div>
      <h1>WebSocket Audio Conversion</h1>
      <button onClick={startRecording}>Start Recording</button>
      <button onClick={stopRecording}>Stop Recording</button>
    </div>
  );
}

export default Audio;
