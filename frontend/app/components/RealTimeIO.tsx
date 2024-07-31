import { useEffect } from "react";
import { sendToRemotePeer } from "~/utils/sendToRemotePeer";

export default function RealTimeIO() {
    useEffect(() => {
        // 最小実装(localhostで動かす場合)はRTCConfigurationは不要
        // 参考：https://blog.ojisan.io/webrtc-video-minimal-impl/
        const pc = new RTCPeerConnection();

        const audioEl = document.getElementById("audio");
        if (audioEl == null || !(audioEl instanceof HTMLAudioElement)) {
            throw new Error("Failed to create audio html element")
        }
        
        // getUserMediaから色々取れる
        // pc.addTrack以外にもカスタマイズの利くpc.addTransceiverがあるらしい
        // 参考：https://zenn.dev/yuki_uchida/books/c0946d19352af5/viewer/320c67
        navigator.mediaDevices
            .getUserMedia({ video: false, audio: { channelCount: 1 } })
            .then(async (stream) => {
                stream.getAudioTracks().forEach(track => pc.addTrack(track, stream));
                // ここじゃなきゃダメか…？
                await handleStartSession(pc);
            })

        pc.ontrack = (event) => {
            audioEl.srcObject = event.streams[0];
        };

        pc.onicecandidate = (event) => {
            if (event.candidate !== null) {
                // candidateに関しては結果をもらわなくても特に問題ないため結果は無視
                // 恐らく result: success が受け取れると思われる
                sendToRemotePeer({
                    type: "candidate",
                    candidate: event.candidate
                })
            }
        };

        const handleStartSession = async (pc: RTCPeerConnection) => {
            const offer = await pc.createOffer();
            await pc.setLocalDescription(offer);
            const response = await sendToRemotePeer({ type: "offer", sdp: offer.sdp });
            const answer = await response.json()
            await pc.setRemoteDescription(answer);
        };

        // handleStartSession();

        return () => {
            if (pc) {
                pc.close();
            }
        };

    }, []);

    return (
        <div>
            <audio id="audio" autoPlay></audio>
        </div>
    )
};
