import React, { useEffect } from 'react';

const RealtimeIOwoSocket: React.FC = () => {
  useEffect(() => {
    const startAudio = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const AudioContext = window.AudioContext || (window as any).webkitAudioContext;
        const audioContext = new AudioContext();
        const source = audioContext.createMediaStreamSource(stream);
        const destination = audioContext.destination;

        source.connect(destination);
      } catch (error) {
        console.error('Error accessing the microphone', error);
      }
    };

    startAudio();
  }, []);

  return (
    <div>
      <h1>マイクからの音声をイヤホンに出力</h1>
    </div>
  );
}

export default RealtimeIOwoSocket;
