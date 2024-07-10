import React, { useState, useEffect, useRef } from 'react';

const ZUNDA_DEFAULT_IMAGE="./zundamon_.png"
const ZUNDA_SELECTED_IMAGE="./zundamon_selected.png"

const Audio: React.FC = () => {
  const [audioContext, setAudioContext] = useState<AudioContext | null>(null);
  const [audioInput, setAudioInput] = useState<MediaStream | null>(null);
  const [socket, setSocket] = useState<WebSocket | null>(null);
  const bufferNumRef = useRef<number>(8);
  const bufferQueueRef = useRef<Float32Array[]>([]);

  useEffect(() => {
    const ws = new WebSocket('ws://192.168.101.200:8765');
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
      const bufferQueue = bufferQueueRef.current;
      const audioData = new Float32Array(event.data.byteLength / 2);
      const view = new DataView(event.data);
      for (let i = 0; i < audioData.length; i++) {
        audioData[i] = view.getInt16(i * 2, true) / 0x7FFF;
      }
      bufferQueue.push(audioData);
      if (bufferQueue.length > 8) {
        bufferQueue.shift();
      }
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

        // バッファのデータを再生
        const outputData = e.outputBuffer.getChannelData(0);
        const bufferQueue = bufferQueueRef.current;
        if (bufferQueue.length > 0) {
          const audioData = bufferQueue.shift();
          if (audioData) {
            for (let i = 0; i < outputData.length; i++) {
              outputData[i] = audioData[i];
            }
          }
        } else {
          for (let i = 0; i < outputData.length; i++) {
            outputData[i] = 0;
          }
        }
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

  const sendValue = (value: number) => {
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ type: 'value', data: value }));
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

    const audioBuffer = context.createBuffer(1, audioData.length, 48000);
    audioBuffer.copyToChannel(audioData, 0);

    const source = context.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(context.destination);
    source.start(0);
  };

  // ================================
  const [currentId, setCurrentId] = useState<number>(1);
  const [id1Selected, setid1Selected] = useState<boolean>(false);
  const [id2Selected, setid2Selected] = useState<boolean>(false);
  const [id3Selected, setid3Selected] = useState<boolean>(false);

  return (
    <div>
      <div id="button-container">
        <img
            id={currentId === 1 ? "selected-button-icon" : "button-icon"}
            onMouseEnter={() => setid1Selected(true)}
            onMouseLeave={() => setid1Selected(false)}
            onClick={() => {setCurrentId(1);sendValue(1);console.log("ID1 Selected!!")}}
            src={id1Selected ? ZUNDA_SELECTED_IMAGE : ZUNDA_DEFAULT_IMAGE}
            alt="id 1"
        />
        <img
            id={currentId === 2 ? "selected-button-icon" : "button-icon"}
            onMouseEnter={() => setid2Selected(true)}
            onMouseLeave={() => setid2Selected(false)}
            onClick={() => {setCurrentId(2);sendValue(2);console.log("ID2 Selected!!")}}
            src={id2Selected ? ZUNDA_SELECTED_IMAGE : ZUNDA_DEFAULT_IMAGE}
            alt="id 2"
        />
        <img
            id={currentId === 3 ? "selected-button-icon" : "button-icon"}
            onMouseEnter={() => setid3Selected(true)}
            onMouseLeave={() => setid3Selected(false)}
            onClick={() => {setCurrentId(3);sendValue(3);console.log("ID3 Selected!!")}}
            src={id3Selected ? ZUNDA_SELECTED_IMAGE : ZUNDA_DEFAULT_IMAGE}
            alt="id 3"
        />
      </div>
      <h1>WebSocket Audio Conversion</h1>
      <button onClick={startRecording}>Start Recording</button>
      <button onClick={stopRecording}>Stop Recording</button>
    </div>
  );
}

export default Audio;
