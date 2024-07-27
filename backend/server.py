import argparse
import asyncio
import json
import logging
import os
import ssl
import uuid

import cv2
from aiohttp import web
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import (MediaBlackhole, MediaPlayer, MediaRecorder,
                                  MediaRelay)
from av import AudioFrame

ROOT = os.path.dirname(__file__)

logger = logging.getLogger("pc")
pcs = set()
relay = MediaRelay()


class AudioTransformTrack(MediaStreamTrack):
    """
    An audio stream track that transforms audio frames from an another track.
    """

    kind = "audio"

    def __init__(self, track):
        super().__init__()
        self.track = track

    async def recv(self):
        frame = await self.track.recv()  # frameはav.AudioFrame型
        new_frame = self.__transform(frame)

        return new_frame

    def __transform(self, frame):
        """
        pts, time_baseはAudioFrameの継承元であるav.frame.Frameのメンバ変数
        フレームの再生順序などを管理している
        ptsが順番を示す数値、time_baseがptsがどのような時間単位で分割しているかを表す
        異なるメディアストリームでの対応(ビデオとオーディオの対応)とかに役立つ
        後はフレームの順序が崩れることを防止したり、空白が生じることを防止する
        他にもこれを利用することで再生速度の変更やシーク操作の実装などが可能らしい

        参考：
        - https://pyav.org/docs/develop/api/audio.html#module-av.audio.frame
        - https://pyav.org/docs/develop/api/frame.html#av.frame.Frame
        """

        npy_frame = frame.to_ndarray() * 2

        new_frame = AudioFrame.from_ndarray(npy_frame, format=frame.format)
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base

        return new_frame


async def offer(request):
    """
    offerは一度のみ行われるシグナリングプロセスの一部
    シグナリングとはremoteとlocalがそれぞれの情報を教えあい、通信経路を確立する感じのものっぽい
    これは一度のみなのでHTTP通信を使っていようがWebRTCの高速性は失われない
    これ以降はremote側でTrackを更新するたびに勝手に通信が走り、やり取りする(ハズ)
    """

    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pc_id = "PeerConnection(%s)" % uuid.uuid4()
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


async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="WebRTC audio / video / data-channels demo"
    )
    parser.add_argument("--cert-file", help="SSL certificate file (for HTTPS)")
    parser.add_argument("--key-file", help="SSL key file (for HTTPS)")
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host for HTTP server (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="Port for HTTP server (default: 8080)"
    )
    parser.add_argument("--record-to", help="Write received media to a file.")
    parser.add_argument("--verbose", "-v", action="count")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.cert_file:
        ssl_context = ssl.SSLContext()
        ssl_context.load_cert_chain(args.cert_file, args.key_file)
    else:
        ssl_context = None

    # Reactをビルドした後であればaiohttpで管理できるかも…？
    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    # app.router.add_get("/", index)
    # app.router.add_get("/client.js", javascript)
    app.router.add_post("/offer", offer)
    web.run_app(
        app, access_log=None, host=args.host, port=args.port, ssl_context=ssl_context
    )
