import argparse
import asyncio
import json
import logging
import os
import uuid

import numpy as np
from aiohttp import web
from aiohttp_cors import ResourceOptions, setup
from aiortc import (MediaStreamTrack, RTCIceCandidate, RTCPeerConnection,
                    RTCSessionDescription)
from aiortc.contrib.media import MediaRelay
from av import AudioFrame

ROOT = os.path.dirname(__file__)

logger = logging.getLogger("pc")
pcs = set()
relay = MediaRelay()

BUFFER_SIZE = 8
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
            async with buffer_lock:
                return buffer.pop(0)
        else:
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

        npy_frame = frame.to_ndarray()

        await asyncio.sleep(1)

        window = np.hanning(npy_frame.shape[0])
        npy_frame = np.multiply(npy_frame, window).astype(np.int16)

        new_frame = AudioFrame.from_ndarray(
            npy_frame, format=frame.format.name, layout="stereo"
        )  # nameまで指定しないとオブジェクトのまま
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        new_frame.sample_rate = frame.sample_rate

        # ========== 調査用 ==========
        # self.__experiment()
        # npy_frame2 = frame.to_ndarray().reshape((2, -1))
        # print(f"Left channel: {npy_frame2[0].shape}")
        # print(f"Right channel: {npy_frame2[1].shape}")
        # print(f"Channels are equal: {np.array_equal(npy_frame2[0], npy_frame2[1])}")
        # # 平均値を取る方法だとエラーはないが、通常よりもかなり低い声で返される
        # mono_frame = np.mean(npy_frame2, axis=0).astype(np.int16).reshape((1, -1))
        # stereo_frame = np.concat([npy_frame[0], npy_frame[0].copy()]).reshape((1, -1))
        # stereo_frame_mix = np.mean(npy_frame2, axis=0).astype(np.int16).reshape((1, -1))
        # new_frame2 = AudioFrame.from_ndarray(
        #     stereo_frame, format=frame.format.name, layout="stereo" # or mono
        # )
        # new_frame2.pts = frame.pts
        # new_frame2.time_base = frame.time_base
        # new_frame2.sample_rate = frame.sample_rate
        # =============================

        async with buffer_lock:
            buffer.append(new_frame)

    def __create_silent_frame(self, frame, stereo=True):
        samples = frame.samples * 2 if stereo else frame.samples
        layout = "stereo" if stereo else "mono"

        silent_data = np.zeros((1, samples), dtype=np.int16)
        silent_frame = AudioFrame.from_ndarray(silent_data, format=frame.format.name, layout=layout)
        
        silent_frame.pts = frame.pts
        silent_frame.time_base = frame.time_base
        silent_frame.sample_rate = frame.sample_rate

        return silent_frame

    def __experiment(self):
        """
        そもそもの問題点として、recvにてself.track.recv()したフレームが持つlayoutがstereoとなってしまっている
        それでもモノラルになっているかもしれないと思ったため調査

        self.track.recv()にて得られるフレーム：
            av.AudioFrame pts=99840, 960 samples at 48000Hz, stereo, s16 at 0x7fe412bd0b20

        ↑ のフレームをnumpyした時のshape:
            (1, 1920)

        関数本文のようにフレームを作成してみると、
            Layout mono :  <av.AudioFrame pts=None, 1920 samples at 0Hz, mono, s16 at 0x7fd6858fb880
            Layout stereo :  <av.AudioFrame pts=None, 960 samples at 0Hz, stereo, s16 at 0x7fd6858fb880

        stereoの方で形状が一致しているのが見て取れる
        実際、コーディックのdefault設定で一度にやり取りされるオーディオの秒数も20msと決まっており、
            48000 * 0.02 = 960
        よりサンプル数も合致していることが分かる

        以上より、最初のTrackから取得する段階でステレオとして認識されていることは間違いなさそう
        ⇒ recvのスクリプトを弄ってmonoにできないか？
        ⇒ 思ったよりも深いところでAudioFrameの設定はされていそう（…というか、サーバー側では形状に関して関知していないのでは…？
        ⇒ ということはクライアント側に問題があるのか？
        ⇒ 確認によればchannelCountは設定できているっぽいからコーデックが悪さをしている…？

        ※ var(track)にて得られた出力：
            {'_events': {}, '_lock': <unlocked _thread.lock object at 0x7f6514ed5000>, '_loop': None, '_waiting': set(), '_MediaStreamTrack__ended': False,
            '_id': '3e24f638-7ca0-43c8-9944-836a43525a2d', 'kind': 'audio', '_relay': <aiortc.contrib.media.MediaRelay object at 0x7f6546fa69b0>,
            '_source': <aiortc.rtcrtpreceiver.RemoteStreamTrack object at 0x7f6514ed8490>, '_buffered': True, '_frame': None,
            '_queue': <Queue at 0x7f6514ed8760 maxsize=0>, '_new_frame_event': None}

        ※ MediaRelayの大元のオブジェクトのrecvメソッド:
            https://github.com/aiortc/aiortc/blob/a9449820f745e63316b57914b2f1fa7c07f54d9a/src/aiortc/rtcrtpreceiver.py#L196

            見たところ、リモート側から送られているフレームを形状とかはそのままに流しているだけっぽい
        """

        a = np.zeros((1, 1920), dtype=np.int16)
        b = np.zeros((1, 1920), dtype=np.int16)
        print("Layout mono : ", AudioFrame.from_ndarray(a, format="s16", layout="mono"))
        print(
            "Layout stereo : ",
            AudioFrame.from_ndarray(b, format="s16", layout="stereo"),
        )


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

        contents = params["candidate"]
        candidate = contents["candidate"]
        if candidate == "":
            # 空のcandidateはICEの収集が終わったことを示す特殊なもの
            # FireFoxの場合、これが送信される(Chromeは暗黙的に無視される)
            # aiortcではparseして自分でオブジェクトを生成する必要がある
            # ⇒ 適当な値を入力しておく(多分大事なのはfoundation="")
            # 参考
            # - https://developer.mozilla.org/en-US/docs/Web/API/RTCIceCandidate/candidate#value
            # - https://stackoverflow.com/questions/67993490/webrtc-empty-string-in-icecandidate
            ice_candidate = RTCIceCandidate(
                component=1,
                foundation="",
                ip="0.0.0.0",
                port="0",
                priority="0",
                protocol="udp",
                type="host",
                sdpMid=contents["sdpMid"],
                sdpMLineIndex=contents["sdpMLineIndex"],
            )
        else:
            candidate_fragments = candidate.split(" ")
            foundation = candidate_fragments[0]
            component = candidate_fragments[1]
            protocol = candidate_fragments[2]
            priority = candidate_fragments[3]
            ip = candidate_fragments[4]
            port = candidate_fragments[5]
            ctype = candidate_fragments[7]
            ice_candidate = RTCIceCandidate(
                foundation=foundation,
                component=component,
                protocol=protocol,
                priority=priority,
                ip=ip,
                port=port,
                type=ctype,
                sdpMid=contents["sdpMid"],
                sdpMLineIndex=contents["sdpMLineIndex"],
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
