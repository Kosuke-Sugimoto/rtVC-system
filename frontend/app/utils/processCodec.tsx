export function setMonoCodec(sdp: string): string {
    const sdpLines = sdp.split('\r\n');

    const monoSupportedCodecs = ['opus', 'G722', 'PCMU', 'PCMA']; // モノラル対応しているコーデックのリスト
    const codecRegex = /^a=rtpmap:(\d+) (\w+)\/(\d+)(?:\/(\d+))?/;

    let validPayloadTypes: string[] = [];
    let removePayloadTypes: string[] = [];

    // SDPの各行をチェックし、モノラル対応コーデックを抽出
    sdpLines.forEach(line => {
        const match = line.match(codecRegex);
        if (match) {
            const payloadType = match[1];
            const codecName = match[2];
            const channels = match[4];

            // モノラルに対応していないコーデックを除去。ただし、Opusは除外
            if (codecName !== 'opus' && (!monoSupportedCodecs.includes(codecName) || (channels && channels !== '1'))) {
                removePayloadTypes.push(payloadType);
            } else {
                validPayloadTypes.push(payloadType);
            }
        }
    });

    // m=audio行の更新
    const mLineIndex = sdpLines.findIndex(line => line.startsWith('m=audio'));
    if (mLineIndex !== -1) {
        const mLineElements = sdpLines[mLineIndex].split(' ');
        const currentPayloadTypes = mLineElements.slice(3);

        // 有効なペイロードタイプのみを残す
        const newPayloadTypes = currentPayloadTypes.filter(pt => validPayloadTypes.includes(pt));

        // m=audio行を更新
        sdpLines[mLineIndex] = [mLineElements[0], mLineElements[1], mLineElements[2], ...newPayloadTypes].join(' ');
    }

    // 無効なコーデックの行を削除
    const updatedSdpLines = sdpLines.filter(line => {
        if (line.startsWith('a=rtpmap:')) {
            const payloadType = line.split(' ')[0].split(':')[1];
            return !removePayloadTypes.includes(payloadType);
        }
        if (line.startsWith('a=fmtp:')) {
            const payloadType = line.split(' ')[0].split(':')[1];
            return !removePayloadTypes.includes(payloadType);
        }
        if (line.startsWith('a=rtcp-fb:')) {
            const payloadType = line.split(' ')[0].split(':')[1];
            return !removePayloadTypes.includes(payloadType);
        }
        return true;
    });

    return updatedSdpLines.join('\r\n');
}

export function setAudioCodec(sdp: string, stereo: boolean): string {
    const sdpLines = sdp.split('\r\n');
    const codecRegex = /^a=rtpmap:(\d+) (\w+)\/(\d+)(?:\/(\d+))?/;
    const opusRegex = /^a=fmtp:(\d+) opus/;

    let opusPayloadType: string | null = null;

    // 有効なペイロードタイプのリスト
    const validPayloadTypes: string[] = [];
    const removePayloadTypes: string[] = [];

    sdpLines.forEach(line => {
        const match = line.match(codecRegex);
        if (match) {
            const payloadType = match[1];
            const codecName = match[2];
            if (codecName === 'opus') {
                opusPayloadType = payloadType;
                validPayloadTypes.push(opusPayloadType);
            } else {
                removePayloadTypes.push(payloadType);
            }
        }
    });

    if (!opusPayloadType) {
        console.error('Opus codec not found in SDP');
        return sdp;
    }

    // SDP行をステレオまたはモノラルに設定
    const updatedSdpLines = sdpLines.map(line => {
        if (line.startsWith(`a=fmtp:${opusPayloadType}`)) {
            return line + (stereo ? ';stereo=1;sprop-stereo=1' : ';stereo=0;sprop-stereo=0');
        }
        return line;
    });

    // 無効なコーデックの行を削除
    const filteredSdpLines = updatedSdpLines.filter(line => {
        if (line.startsWith('a=rtpmap:') || line.startsWith('a=fmtp:') || line.startsWith('a=rtcp-fb:')) {
            const payloadType = line.split(' ')[0].split(':')[1];
            return !removePayloadTypes.includes(payloadType);
        }
        return true;
    });

    // m=audio行の更新
    const mLineIndex = filteredSdpLines.findIndex(line => line.startsWith('m=audio'));
    if (mLineIndex !== -1) {
        const mLineElements = filteredSdpLines[mLineIndex].split(' ');
        const currentPayloadTypes = mLineElements.slice(3);

        // Opusコーデックのペイロードタイプのみを残す
        const newPayloadTypes = currentPayloadTypes.filter(pt => pt === opusPayloadType);
        filteredSdpLines[mLineIndex] = [mLineElements[0], mLineElements[1], mLineElements[2], ...newPayloadTypes].join(' ');
    }

    return filteredSdpLines.join('\r\n');
}

export function preferCodec(sdp: string, codec: string): string {
    const sdpLines = sdp.split('\r\n');
    const mLineIndex = sdpLines.findIndex(line => line.startsWith('m=audio'));
    if (mLineIndex === -1) {
        return sdp;
    }

    const codecPayloadTypes: string[] = [];
    const codecRegex = new RegExp(`a=rtpmap:(\\d+) ${codec}/\\d+`);

    for (let i = 0; i < sdpLines.length; i++) {
        const match = sdpLines[i].match(codecRegex);
        if (match) {
            codecPayloadTypes.push(match[1]);
        }
    }

    if (codecPayloadTypes.length > 0) {
        const mLineElements = sdpLines[mLineIndex].split(' ');
        const currentPayloadTypes = mLineElements.slice(3);

        // コーデックの優先順序を新しい順序で再構築
        const newPayloadTypes = codecPayloadTypes.concat(
            currentPayloadTypes.filter(pt => !codecPayloadTypes.includes(pt))
        );

        // m=audio行を更新
        sdpLines[mLineIndex] = [mLineElements[0], mLineElements[1], mLineElements[2], ...newPayloadTypes].join(' ');
    }

    return sdpLines.join('\r\n');
}

export function modifyOpusParams(sdp: string, bitrate: number, channels: number): string {
    // Opusに関する行を検索し、ビットレート、CBR、モノラルを設定
    sdp = sdp.replace(/a=fmtp:111 .*\r\n/g, `a=fmtp:111 minptime=10; useinbandfec=1; maxaveragebitrate=${bitrate}; cbr=1; stereo=${channels === 2 ? 1 : 0}\r\n`);
    return sdp;
}

// 結合された関数
export function setOpusCBRMono(sdp: string, bitrate: number): string {
    // まず、モノラル対応のコーデックを設定。ただし、Opusは除外
    sdp = setMonoCodec(sdp);

    // 次に、Opusコーデックのビットレートとモノラル設定を行う
    sdp = modifyOpusParams(sdp, bitrate, 1);

    return sdp;
}
