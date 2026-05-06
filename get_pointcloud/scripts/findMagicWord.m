%% MARS UART Point Cloud Capture
clear; clc; close all;

%% ============================================================
%  設定區
%% ============================================================
CONFIG_PORT  = 'COM7';
DATA_PORT    = 'COM8';
CONFIG_FILE  = 'xwr68xx_MARS_UART_test.cfg';
N_FRAMES     = 200;
MAX_POINTS   = 64;
SAVE_FILE    = 'mars_pointcloud_0506_right_limb_extension.mat';

%% ============================================================
%  TLV 常數
%% ============================================================
TLV_DETECTED_POINTS = 1;
TLV_SIDE_INFO       = 7;
MAGIC_WORD = uint8([2 1 4 3 6 5 8 7]);

%% ============================================================
%  開啟 Serial Ports
%% ============================================================
fprintf('[INFO] 開啟 Serial ports...\n');
cfgPort  = serialport(CONFIG_PORT, 115200);
dataPort = serialport(DATA_PORT,   921600);
configureTerminator(cfgPort, "LF");

%% ============================================================
%  送出 Config
%% ============================================================
fprintf('[INFO] 送出 config...\n');
fid = fopen(CONFIG_FILE, 'r');
while ~feof(fid)
    line = strtrim(fgetl(fid));
    if isempty(line) || line(1) == '%'
        continue;
    end
    writeline(cfgPort, line);
    pause(0.05);
    fprintf('  [CFG] %s\n', line);
end
fclose(fid);
fprintf('[INFO] Config 送出完成。\n\n');

%% ============================================================
%  預先分配儲存空間
%% ============================================================
marsData = zeros(N_FRAMES, MAX_POINTS, 5, 'single');

%% ============================================================
%  建立即時視覺化
%% ============================================================
fig = figure('Name','MARS 即時點雲','NumberTitle','off', ...
             'Position',[100 100 900 700]);
ax = axes(fig);
hold(ax, 'on');

h = scatter3(ax, zeros(64,1), zeros(64,1), zeros(64,1), ...
             36, zeros(64,1), 'filled');

xlabel(ax, 'X (m)');
ylabel(ax, 'Y (m)');
zlabel(ax, 'Z (m)');
xlim(ax, [-1 1]);
ylim(ax, [0 3]);
zlim(ax, [-1 1]);
grid(ax, 'on');
view(ax, 45, 30);
colorbar(ax);
clim(ax, [0 50]);
title(ax, '等待資料...');

%% ============================================================
%  主迴圈
%% ============================================================
fprintf('[INFO] 開始錄製，目標 %d frames。\n', N_FRAMES);
fprintf('[INFO] 請站在雷達前方 1~2 公尺處。\n\n');

byteBuffer = uint8([]);
frameCount = 0;

while frameCount < N_FRAMES && ishandle(fig)

    % 讀取新 bytes
    nAvail = dataPort.NumBytesAvailable;
    if nAvail > 0
        newBytes = read(dataPort, nAvail, 'uint8');
        byteBuffer = [byteBuffer, newBytes];
    end

    % 防止 buffer 無限增長
    if length(byteBuffer) > 65536
        byteBuffer = byteBuffer(end-32767:end);
    end

    % 找 magic word
    magicIdx = findMagicWord(byteBuffer, MAGIC_WORD);
    if isempty(magicIdx)
        pause(0.005);
        continue;
    end

    % 確認 frame 長度欄位已到達
    if length(byteBuffer) < magicIdx + 15
        pause(0.005);
        continue;
    end

    % 讀取 totalPacketLen
    totalLen = typecast(byteBuffer(magicIdx+12 : magicIdx+15), 'uint32');

    % 確認完整 frame 已到達
    if length(byteBuffer) < magicIdx - 1 + totalLen
        pause(0.005);
        continue;
    end

    % 擷取並解析 frame
    frameBytes = byteBuffer(magicIdx : magicIdx-1+totalLen);
    byteBuffer = byteBuffer(magicIdx+totalLen : end);

    rawPoints = parseFrame(frameBytes, TLV_DETECTED_POINTS, TLV_SIDE_INFO);

    if isempty(rawPoints)
        continue;
    end
    

    % 轉成 MARS 格式
    marsFrame = toMARSFormat(rawPoints, MAX_POINTS);

    % 儲存
    frameCount = frameCount + 1;
    marsData(frameCount, :, :) = marsFrame;

    % 顯示進度
    validIdx = any(marsFrame ~= 0, 2);
    nPts = sum(validIdx);
    fprintf('Frame %3d/%d | 有效點數: %2d\n', frameCount, N_FRAMES, nPts);

    % 更新視覺化
    if ishandle(fig)
        set(h, 'XData', marsFrame(:,1), ...
               'YData', marsFrame(:,2), ...
               'ZData', marsFrame(:,3), ...
               'CData', marsFrame(:,5));
        title(ax, sprintf('Frame %d/%d | 有效點數：%d', ...
              frameCount, N_FRAMES, nPts));
        drawnow limitrate;
    end

end

%% ============================================================
%  關閉 ports 並儲存
%% ============================================================
clear cfgPort dataPort;
fprintf('\n[INFO] 錄製完成，共 %d frames。\n', frameCount);

marsData = marsData(1:frameCount, :, :);
save(SAVE_FILE, 'marsData');
fprintf('[INFO] 已儲存至 %s\n', SAVE_FILE);
fprintf('[INFO] 資料維度：[%d × %d × 5]\n', frameCount, MAX_POINTS);

validCounts = squeeze(sum(any(marsData ~= 0, 3), 2));
fprintf('[INFO] 平均有效點數：%.1f\n', mean(validCounts));
fprintf('[INFO] 最大有效點數：%d\n',   max(validCounts));
fprintf('[INFO] 最小有效點數：%d\n',   min(validCounts));