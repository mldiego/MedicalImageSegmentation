
rng(0);
aa = rand(10)-0.1;
bb = aa + 0.1;
cc = ImageStar(aa,bb);
layer = ReluLayer;
xx = layer.reach_star_single_input(cc, 'relax-star-range', 1, [], 'linprog');

[l,u] = xx.estimateRanges;
xx2 = ImageStar(l,u);

[m1,M1] = xx.getRanges;
[m2,M2] = xx2.getRanges;

% Is xx3 same as xx2?
% Get dimensions
h = xx.height;
w = xx.width;
c = xx.numChannel;

% create star
center = 0.5*(u+l);
v = 0.5*(u-l);
n = numel(center);
V = zeros(h,w,c,n, 'like', center);
bCount = 1;
for i = 1:size(v,1)
    for j = 1:size(v,2)
        V(i,j,:,bCount) = v(i,j);
        bCount = bCount + 1;
    end
end
C = zeros(1, n, 'like', V);
d = zeros(1, 1, 'like', V);
Y = ImageStar(cat(4,center,V), C, d, -1*ones(n,1), ones(n,1), l, u);

[m3,M3] = Y.estimateRanges;

% Looks good

% Can we estimate ranges faster now?

f = Y.V;

x1 = f(:,:,:,1) + tensorprod(f(:,:,:,2:end),Y.pred_lb, [3,4], [2,1]);
x2 = f(:,:,:,1) + tensorprod(f(:,:,:,2:end),Y.pred_ub, [3,4], [2,1]);
m4 = min(x1,x2);
M4 = max(x1,x2);

% Yes we can
