function marsFrame = toMARSFormat(points, maxPoints)
    if nargin < 2; maxPoints = 64; end
    marsFrame = zeros(maxPoints, 5, 'single');
    if isempty(points) || size(points,1) == 0; return; end
    inRange = points(:,1)>=-1 & points(:,1)<=1 & ...
              points(:,2)>= 0 & points(:,2)<= 3 & ...
              points(:,3)>=-1 & points(:,3)<= 1;
    inRangePts  = points( inRange,:);
    outRangePts = points(~inRange,:);
    nIn  = size(inRangePts, 1);
    nOut = size(outRangePts,1);
    if nIn >= maxPoints
        [~,idx] = sort(inRangePts(:,5),'descend');
        points = inRangePts(idx(1:maxPoints),:);
    elseif nIn+nOut > maxPoints
        nOutKeep = maxPoints-nIn;
        [~,idx] = sort(outRangePts(:,5),'descend');
        points = [inRangePts; outRangePts(idx(1:nOutKeep),:)];
    else
        points = [inRangePts; outRangePts];
    end
    N = size(points,1);
    points = sortrows(points,[1,2,3]);
    marsFrame(1:N,:) = single(points);
end
