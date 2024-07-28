type CandidateProps = {
    type: string,
    candidate: RTCIceCandidate
};

type OfferProps = {
    type: string,
    sdp: string | undefined
}

export const sendToRemotePeer = async (message: CandidateProps | OfferProps) => {
    const response = await fetch(
        "http://localhost:8080/offer",
        {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(message)
        }
    );
    return response;
};
