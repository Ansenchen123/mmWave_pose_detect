function points = parseFrame(data, TLV_POINTS, TLV_SIDE)
% 解析一個完整 frame，回傳 [N x 5]
% 欄位：[x, y, z, doppler, SNR]

    points = [];
    offset = 9;  % 跳過 magic word (8 bytes)

    try
        offset = offset + 4;  % version
        offset = offset + 4;  % totalLen
        offset = offset + 4;  % platform
        offset = offset + 4;  % frameNumber
        offset = offset + 4;  % timeCPUCycles

        numDetObj = typecast(data(offset:offset+3), 'uint32');
        offset = offset + 4;

        numTLV = typecast(data(offset:offset+3), 'uint32');
        offset = offset + 4;

        offset = offset + 4;  % subFrameNumber
    catch
        return;
    end

    if numDetObj == 0
        points = zeros(0, 5);
        return;
    end

    points = zeros(numDetObj, 5);

    for t = 1:numTLV
        if offset + 7 > length(data); break; end

        tlvType   = typecast(data(offset:offset+3), 'uint32'); offset = offset+4;
        tlvLength = typecast(data(offset:offset+3), 'uint32'); offset = offset+4;

        if tlvType == TLV_POINTS
            for i = 1:numDetObj
                if offset+15 > length(data); break; end
                points(i,1) = typecast(data(offset:offset+3),'single'); offset=offset+4;
                points(i,2) = typecast(data(offset:offset+3),'single'); offset=offset+4;
                points(i,3) = typecast(data(offset:offset+3),'single'); offset=offset+4;
                points(i,4) = typecast(data(offset:offset+3),'single'); offset=offset+4;
            end

        elseif tlvType == TLV_SIDE
            for i = 1:numDetObj
                if offset+3 > length(data); break; end
                snr = typecast(data(offset:offset+1),'uint16'); offset=offset+2;
                offset = offset+2;  % skip noise
                points(i,5) = double(snr) * 0.1;  % 換算成 dB
            end

        else
            offset = offset + tlvLength;
        end
    end
end