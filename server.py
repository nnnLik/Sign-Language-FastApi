import asyncio
import os

from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaRelay

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBasic
from fastapi.responses import JSONResponse

from starlette.requests import Request
from starlette.responses import HTMLResponse
from starlette.templating import Jinja2Templates

from service.sign_track import Sign
from schemas.offer import Offer

ROOT = os.path.dirname(__file__)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

webcam = None
new_video_track = None
pcs = set()
dcs = set()
relay = MediaRelay()

security = HTTPBasic()
app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health", include_in_schema=False)
def health() -> JSONResponse:
    """ヘルスチェック"""
    return JSONResponse({"message": "It worked!!"})

@app.post("/offer")
async def offer(params: Offer):
    offer = RTCSessionDescription(sdp=params.sdp, type=params.type)

    pc = RTCPeerConnection()
    pcs.add(pc)

    recorder = MediaBlackhole()


    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    @pc.on("track")
    def on_track(track):
        global new_video_track
        new_video_track = Sign(relay.subscribe(track))
        pc.addTrack(new_video_track)

        @track.on("ended")
        async def on_ended():
            await recorder.stop()

    @pc.on("datachannel")
    def on_datachannel(channel):
        global new_video_track
        new_video_track.channel = channel

        @channel.on("message")
        async def on_message(message):
            if isinstance(message, str):
                data = message.encode("utf-8")
            else:
                data = message

    # handle offer
    await pc.setRemoteDescription(offer)
    await recorder.start()

    # send answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return JSONResponse(
        {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type},
    )

@app.on_event("shutdown")
async def on_shutdown():
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()