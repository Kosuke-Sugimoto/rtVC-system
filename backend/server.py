import argparse
import asyncio
import json
import logging
import os
import uuid

import numpy as np
import soxr
import torch
from aiohttp import web
from aiohttp_cors import ResourceOptions, setup
from aiortc import (MediaStreamTrack, RTCIceCandidate, RTCPeerConnection,
                    RTCSessionDescription)
from aiortc.contrib.media import MediaRelay
from av import AudioFrame
from frcrn import denoise, initialize_frcrn
from hifigan_fix.inference_rt import get_mel_torch, inference_hg, initialize_hg
from starganv2_vc.inference_rt import conversion, initialize_vc

ROOT = os.path.dirname(__file__)

logger = logging.getLogger("pc")
pcs = set()
relay = MediaRelay()

MAX_WAV_VALUE = 32768
BUFFER_SIZE = 2
buffer = []
buffer_lock = asyncio.Lock()
background_tasks = set()


class AudioTransformTrack(MediaStreamTrack):
    """
    An audio stream track that transforms audio frames from an another track.
    """

    kind = "audio"

    def __init__(self, track):
        super().__init__()
        self.track = track
        self.inout_samplerate = 48000
        self.denoise_samplerate = 16000
        self.inter_samplerate = 24000
        h, device = initialize_hg("config_v1_mod_2.json", "g_07180000_2")
        initialize_vc(
            h,
            device,
            "ita4jvs20_pre_alljp",
            "epoch_00294.pth",
            "ep50_200bat32lr5_alljp.pth",
            "model",
        )
        self.stft_factor = 300
        self.down_factor = 2
        self.inter_size_vc = 80
        self.inout_size_vc = self.inter_size_vc * self.stft_factor * self.down_factor
        initialize_frcrn(
            device,
            int(self.inout_size_vc / self.inout_samplerate * self.denoise_samplerate),
        )
        self.nwarmup = 50
        self.__warmup()

    def __warmup(self):
        print("Warm up...")
        ref_emb_key = "zundamon127"
        with torch.no_grad():
            for _ in range(self.nwarmup):
                input_wave = np.random.random_sample((self.inout_size_vc,)).astype(
                    np.float32
                )
                input_wave = soxr.resample(
                    input_wave, self.inout_samplerate, self.denoise_samplerate, "VHQ"
                )
                input_wave = denoise(input_wave)
                input_wave = soxr.resample(
                    input_wave, self.denoise_samplerate, self.inter_samplerate, "VHQ"
                )
                input_mel = get_mel_torch(input_wave[None])
                output_mel = conversion(input_mel, ref_emb_key)
                output_wave = inference_hg(output_mel).cpu().numpy()[0]
                output_wave = soxr.resample(
                    output_wave, self.inter_samplerate, self.inout_samplerate, "VHQ"
                )
        print("Done.")

    async def recv(self):
        """
        変換処理に時間がかかるため、メインでバッファへの登録をしつつ、裏で変換処理を回したい
        　⇒変換処理を非同期処理にする & メインでawaitをさせない
        　⇒変換が終わり次第バッファに登録するようにし、そのバッファの長さが規定値以上なら読み取り始める
        　⇒それまでは無音のフレームを返しておく

        以上のような機能(非同期処理の結果を待たないで処理を継続)は"fire and forget"と呼ばれている
        Fire and Forgetについて
        参考：
        - https://qiita.com/eycjur/items/5e8df3549f6c069429dd#%E9%9D%9E%E5%90%8C%E6%9C%9F%E9%96%A2%E6%95%B0%E3%81%8B%E3%82%89%E9%9D%9E%E5%90%8C%E6%9C%9F%E9%96%A2%E6%95%B0%E3%82%92%E5%91%BC%E3%81%B6
          ここにある通り、create_taskを使うことでタスクをスケジュール化(バックグラウンドに回す)できる
        - https://qiita.com/eycjur/items/5e8df3549f6c069429dd#%E5%86%8D%E8%80%83%E5%90%8C%E6%9C%9F%E9%96%A2%E6%95%B0%E3%81%8B%E3%82%89%E5%90%8C%E6%9C%9F%E9%96%A2%E6%95%B0%E3%82%92%E5%91%BC%E3%81%B6fire-and-forget
          同様の記事だが、ここでは'asyncio.new_event_loop().run_in_executor(None, 任意のタスク)'のように紹介されている
          - 既にイベントループが存在する場合、新たに作成する必要はない(今回が該当)
          - run_in_executorは同期処理の関数を非同期的に処理するためのメソッド(今回は不適当)
            - 別スレッドで処理するためのものらしい
        - https://docs.python.org/ja/3.10/library/asyncio-task.html#asyncio.create_task
          公式にもcreate_taskは暗黙的にfire-and-forgetに使うものとしているみたい
          ここに書いてある通り、完了していなくてもガベージコレクションされる恐れがあるっぽい
          ⇒今回遅延が入りすぎると消されちゃう…？(とりまsetで参照は持たせておく)

        バッファへの書き込みと読み込みが別スレッドで存在する以上、とりあえずlockはかけておく
        参考：https://docs.python.org/ja/3/library/asyncio-sync.html#lock
        """

        frame = await self.track.recv()  # frameはav.AudioFrame型

        task = asyncio.create_task(self.__transform(frame))

        background_tasks.add(task)
        task.add_done_callback(background_tasks.discard)

        if len(buffer) >= BUFFER_SIZE:
            print("vc")
            async with buffer_lock:
                return buffer.pop(0)
        else:
            print("sil")
            return self.__create_silent_frame(frame)

    async def __transform(self, frame):
        """
        pts, time_baseはAudioFrameの継承元であるav.frame.Frameのメンバ変数
        フレームの再生順序などを管理している
        ptsが順番を示す数値、time_baseがptsがどのような時間単位で分割しているかを表す
        異なるメディアストリームでの対応(ビデオとオーディオの対応)とかに役立つ
        後はフレームの順序が崩れることを防止したり、空白が生じることを防止する
        他にもこれを利用することで再生速度の変更やシーク操作の実装などが可能らしい

        new_frame作成の際にはsample_rateを引き継がせておかないと以下のようなエラーが発生
        [auto_aresample_0 @ 0x7f91f0004f40] [SWR @ 0x7f91f00050c0] Requested input sample rate 0 is invalid
        [auto_aresample_0 @ 0x7f91f0004f40] Failed to configure output pad on auto_aresample_0
        WARNING:aiortc.rtcrtpsender:RTCRtpsender(audio) Traceback (most recent call last):
        File "/usr/local/lib/python3.10/dist-packages/aiortc/rtcrtpsender.py", line 346, in _run_rtp
            enc_frame = await self._next_encoded_frame(codec)
        File "/usr/local/lib/python3.10/dist-packages/aiortc/rtcrtpsender.py", line 293, in _next_encoded_frame
            payloads, timestamp = await self.__loop.run_in_executor(
        File "/usr/lib/python3.10/concurrent/futures/thread.py", line 58, in run
            result = self.fn(*self.args, **self.kwargs)
        File "/usr/local/lib/python3.10/dist-packages/aiortc/codecs/opus.py", line 82, in encode
            for frame in self.resampler.resample(frame):
        File "av/audio/resampler.pyx", line 34, in av.audio.resampler.AudioResampler.resample
        File "av/audio/resampler.pyx", line 90, in av.audio.resampler.AudioResampler.resample
        File "av/filter/graph.pyx", line 42, in av.filter.graph.Graph.configure
        File "av/error.pyx", line 326, in av.error.err_check
        av.error.ValueError: [Errno 22] Invalid argument

        参考：
        - https://pyav.org/docs/develop/api/audio.html#module-av.audio.frame
        - https://pyav.org/docs/develop/api/frame.html#av.frame.Frame
        """

        npy_frame = frame.to_ndarray().astype(np.float32)

        if self.inout_samplerate != frame.sample_rate:
            self.inout_samplerate = frame.sample_rate

        converted_npy_frame = self.__conversion(npy_frame)

        # await asyncio.sleep(1)

        window = np.hanning(converted_npy_frame.shape[0])
        converted_npy_frame = np.multiply(converted_npy_frame, window).astype(np.int16)

        new_frame = AudioFrame.from_ndarray(
            converted_npy_frame, format=frame.format.name
        )  # nameまで指定しないとオブジェクトのまま
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        new_frame.sample_rate = frame.sample_rate

        async with buffer_lock:
            buffer.append(new_frame)

    def __create_silent_frame(self, frame):
        # npy_frameは(1, 1920)、つまり、(1, sample数×channel数)
        silent_data = np.zeros((1, frame.samples), dtype=np.int16)
        silent_frame = AudioFrame.from_ndarray(silent_data, format=frame.format.name)
        silent_frame.pts = frame.pts
        silent_frame.time_base = frame.time_base
        silent_frame.sample_rate = frame.sample_rate

        return silent_frame

    def __conversion(self, audio_data):
        # リサンプリング
        # audio_dataのshape：(1, 1920)
        # soxr.resampleの要求：1D(mono) or 2D(frames, channels) array input
        audio_data = audio_data.squeeze(axis=0)
        audio_data /= MAX_WAV_VALUE

        default_len = audio_data.shape[0]
        wrapper_data = np.zeros(self.inout_size_vc, np.float32)
        wrapper_data[:default_len] = audio_data
        input_wave = soxr.resample(
            wrapper_data, self.inout_samplerate, self.denoise_samplerate, "VHQ"
        )

        # ノイズ除去
        input_wave = denoise(input_wave)

        # input_wave /= MAX_WAV_VALUE

        # 再リサンプリング
        input_wave = soxr.resample(
            input_wave, self.denoise_samplerate, self.inter_samplerate, "VHQ"
        )

        # メルスペクトログラムに変換
        input_mel = get_mel_torch(input_wave[None])

        # 声質変換
        # ref_emb_key = 'zundamon127'
        # output_mel = conversion(input_mel, ref_emb_key)

        # 最終的な音声生成
        # output_wave = inference_hg(output_mel).cpu().detach().numpy()[0]
        output_wave = inference_hg(input_mel).cpu().detach().numpy()[0]
        output_wave = output_wave[:default_len]

        # リサンプリング
        converted_audio_data = soxr.resample(
            output_wave, self.inter_samplerate, self.inout_samplerate, "VHQ"
        )
        converted_audio_data = np.expand_dims(
            converted_audio_data, axis=0
        )  # = torch.unsqeeze
        converted_audio_data *= MAX_WAV_VALUE
        converted_audio_data = converted_audio_data.astype(np.int16)

        # vad_threshold = 0.00005
        # if np.average(np.power(input_wave[:default_len], 2)) < vad_threshold:
        #     converted_audio_data.fill(0)

        return converted_audio_data


async def offer(request):
    """
    offerは一度のみ行われるシグナリングプロセスの一部
    シグナリングとはremoteとlocalがそれぞれの情報を教えあい、通信経路を確立する感じのものっぽい
    これは一度のみなのでHTTP通信を使っていようがWebRTCの高速性は失われない
    これ以降はremote側でTrackを更新するたびに勝手に通信が走り、やり取りする(ハズ)
    """

    params = await request.json()
    pc_id = "PeerConnection(%s)" % uuid.uuid4()
    logger.info("Received %s from %s", params["type"], request.remote)

    if params["type"] == "offer":
        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

        pc = RTCPeerConnection()
        pcs.add(pc)

        def log_info(msg, *args):
            logger.info(pc_id + " " + msg, *args)

        log_info("Created for %s", request.remote)

        # MediaPlayerはファイルなどからビデオ・オーディオを読み取る関数
        #  ファイル、URL、ウェブカメラ等から読み取れるっぽい
        #  参考：https://aiortc.readthedocs.io/en/latest/helpers.html#aiortc.contrib.media.MediaPlayer
        # MediaRecorderはビデオ・オーディオをファイルに書き込む関数
        #  参考：https://aiortc.readthedocs.io/en/latest/helpers.html#aiortc.contrib.media.MediaRecorder
        # MediaBlackholeはソースを受け取って破棄するためのもの(例外処理用…？)
        #  参考：https://aiortc.readthedocs.io/en/latest/helpers.html#aiortc.contrib.media.MediaBlackhole
        # Playerは'media source'、Recorderは'media sink'と書かれていたがこれはいったい…？

        # TODO : 要らなさそうなら消す
        # player = MediaPlayer(os.path.join(ROOT, "demo-instruct.wav"))
        # if args.record_to:
        #     recorder = MediaRecorder(args.record_to)
        # else:
        #     recorder = MediaBlackhole()

        # pc.on(...)はイベントリスナー、callback関数を定義しそれを実行する
        @pc.on("datachannel")
        def on_datachannel(channel):
            @channel.on("message")
            def on_message(message):
                if isinstance(message, str) and message.startswith("ping"):
                    channel.send("pong" + message[4:])

        @pc.on("icecandidate")
        def on_icecandidate(event):
            """
            aiortcライブラリが内部で管理するみたいなのでサーバー側では明示的に指定する必要はない
            ログの設定をしたり処理の追加をする際にはこのように設定が可能

            何やらICEに関してConnecting...のまま推移しない問題も発生するみたい
            https://github.com/aiortc/aiortc/issues/1084
            一応デフォルトのままやれば問題なさそうだが、細かく設定するとダメなのかな…？(読んだ当時では知識不足)

            Connecting...のまま推移しない問題が発生したが、原因としてはクライアント側でsetRemoteDescriptionしていなかったことだった模様
            """

            if event.candidate:
                log_info("ICE Candidate: %s", event.candidate)

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            log_info("Connection state is %s", pc.connectionState)
            if pc.connectionState == "failed":
                await pc.close()
                pcs.discard(pc)

        @pc.on("track")
        def on_track(track):
            """
            relay.subscribeはトラックを複製するためのメソッド
            複数クライアント・複数種の異なる変換処理を施したいときに役立つ
            　⇒YouTubeとかの配信みたいな…？
            クライアントとサーバーが1対1なら必ずしも必要ないような…？
            参考：https://aiortc.readthedocs.io/en/latest/helpers.html#aiortc.contrib.media.MediaRelay
            """

            log_info("Track %s received", track.kind)

            if track.kind == "audio":
                pc.addTrack(AudioTransformTrack(relay.subscribe(track)))

            # TODO：要らなさそうなら消す
            # @track.on("ended")
            # async def on_ended():
            #     log_info("Track %s ended", track.kind)
            #     await recorder.stop()

        # SDP(Session Description Protocol)を設定する
        # remote: client, local: server (server.py視点)
        # オファーをremoteから受け取った際にはremoteの設定？をlocalに登録するそうな
        await pc.setRemoteDescription(offer)

        # オファーに対する回答(Answer SDP)を作成
        # localで生成した回答をremoteに登録(注意!!：登録しただけで送信はしていない)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        # remoteに登録した回答(Answer 'SDP')をremoteに送信する
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
            ),
        )

    elif params["type"] == "candidate":
        # paramsの中身一例
        # {'type': 'candidate', 'candidate': {'candidate': 'candidate:244898170 1 tcp 1518217471 ...(ip) 9 typ host tcptype active generation 0 ufrag QBvn network-id 3', 'sdpMid': '0', 'sdpMLineIndex': 0, 'usernameFragment': 'QBvn'}}
        # ローカルから取得したcandidate情報をそのまま入力してもタイプ不一致でエラーを吐かれる
        # そのためRTCIceCandidateクラスでラップする必要がある
        # 参考：
        # - https://aiortc.readthedocs.io/en/latest/api.html#aiortc.RTCIceCandidate
        # - https://developer.mozilla.org/en-US/docs/Web/API/RTCIceCandidate/RTCIceCandidate
        # - https://github.com/aiortc/aiortc/issues/1084#:~:text=async%20def%20handle_candidate(,append(rtc_candidate)
        candidate = params["candidate"]
        foundation = candidate["candidate"].split(" ")[0]
        component = candidate["candidate"].split(" ")[1]
        protocol = candidate["candidate"].split(" ")[2]
        priority = candidate["candidate"].split(" ")[3]
        ip = candidate["candidate"].split(" ")[4]
        port = candidate["candidate"].split(" ")[5]
        ctype = candidate["candidate"].split(" ")[7]
        ice_candidate = RTCIceCandidate(
            foundation=foundation,
            component=component,
            protocol=protocol,
            priority=priority,
            ip=ip,
            port=port,
            type=ctype,
            sdpMid=candidate["sdpMid"],
            sdpMLineIndex=candidate["sdpMLineIndex"],
        )
        for pc in pcs:
            await pc.addIceCandidate(ice_candidate)
        return web.Response(
            content_type="application/json", text=json.dumps({"result": "success"})
        )

    else:
        return web.Response(status=400, text="Invalid request type")


async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WebRTC Server")
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host for HTTP server (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="Port for HTTP server (default: 8080)"
    )
    parser.add_argument("--verbose", "-v", action="count")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # Reactをビルドした後であればaiohttpで管理できるかも…？
    app = web.Application()

    app.on_shutdown.append(on_shutdown)

    # app.router.add_get("/", index)
    # app.router.add_get("/client.js", javascript)
    app.router.add_post("/offer", offer)

    # 別ソースからのアクセスだとcorsでエラーを吐かれるのでここで設定
    # no-corsモードは制限が多いらしい(未調査)
    cors = setup(
        app,
        defaults={
            "*": ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
            )
        },
    )
    for route in list(app.router.routes()):
        cors.add(route)

    web.run_app(app, access_log=None, host=args.host, port=args.port)
