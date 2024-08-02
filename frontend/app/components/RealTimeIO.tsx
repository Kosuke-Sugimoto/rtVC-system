import { useEffect } from "react";
import { preferCodec, setAudioCodec, setMonoCodec } from "~/utils/processCodec";
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
            .getUserMedia({
                video: false,
                audio: {
                    channelCount: { exact: 1 },
                    sampleRate: { exact: 48000 }, // FireFoxはサポート外
                    sampleSize: { exact: 16 } // FireFoxはサポート外
                    // exactを入れることで指定した値にマッチングするようになる？
                    // 参考
                    // - https://developer.mozilla.org/en-US/docs/Web/API/Media_Capture_and_Streams_API/Constraints#specifying_a_range_of_values
                    // - https://developer.mozilla.org/en-US/docs/Web/API/MediaTrackSettings/sampleSize
                    // - https://developer.mozilla.org/en-US/docs/Web/API/MediaTrackSettings/sampleRate
                }
            })
            .then(async (stream) => {
                stream.getAudioTracks().forEach(track => pc.addTrack(track, stream));
                console.log("Current Setting is below :");
                console.log(stream.getAudioTracks()[0].getSettings()); // 設定が適用できているのか確認
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
        
        // やはり、AudioFrameでmonoにならなかったのはOpusだった模様
        // 以下のようにmono非対応のコーデックを全消去することで無理矢理monoにすることは可能
        // offer.sdp = setMonoCodec(offer.sdp!);
        // が、以下に示すサーバー側の出力を見て分かる通り、音質に難がありすぎる
        // <av.AudioFrame pts=24960, 160 samples at 8000Hz, mono, s16 at 0x7fae0e5f63e0
        // そもそもOpusはmonoもできるはずだが、なぜか指定しても反応してくれない模様
        const handleStartSession = async (pc: RTCPeerConnection) => {
            const offer = await pc.createOffer();
            console.log(offer.sdp);
            // offer.sdp = setAudioCodec(offer.sdp!, false);
            // console.log(offer.sdp);
            offer.sdp = setMonoCodec(offer.sdp!);
            console.log(offer.sdp);
            offer.sdp = preferCodec(offer.sdp, "G722")
            console.log(offer.sdp);
            await pc.setLocalDescription(offer);
            const response = await sendToRemotePeer({ type: "offer", sdp: offer.sdp });
            const answer = await response.json()
            await pc.setRemoteDescription(answer);
        };

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
