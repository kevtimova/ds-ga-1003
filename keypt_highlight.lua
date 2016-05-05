-- load required modules
require 'image'
require 'nn'
require 'math'
require 'torch'
require 'io'
stringx = require('pl.stringx')

-- adjust size of neighborhood to preserve around each key point
scale = 2

-- function calculating neightborhood bounds
function neighborhood(keypt)
	local range = math.floor(keypt[3]*scale)
	local x1 = math.max(keypt[1]-range, 1)
	local x2 = math.min(keypt[1]+range,200)
	local y1 = math.max(keypt[2]-range, 1)
	local y2 = math.min(keypt[2]+range,200)
	return x1, x2, y1, y2
end

-- function returning only key point neighborhoods of a given image (and its index in pts)
function image_kpts(img, kpt_ind)
    local img_copy = torch.Tensor():typeAs(img):resizeAs(img):fill(0)
    for j = 1, #pts[kpt_ind] do
        x1, x2, y1, y2 = neighborhood(pts[kpt_ind][j])
        for d = 1,img:size(1) do
            img_copy[{{d},{x1,x2},{y1,y2}}] = img[{{d},{x1,x2},{y1,y2}}]
        end
    end
    return img_copy
end

-- load original 200x200 images
images = torch.load("data/5_expression_data.t7")

-- open key points file
keypoints = io.open("data/5_expression_kpt.txt", 'r')
io.input(keypoints)

-- store image key points into a nested table 
pts = {}
while true do
	-- read a line from the key points file
    line = io.read()
    if line == nil then
        break
    end
    -- transform a line into a table of numbers
    line = stringx.split(line, ",")
    for j = 1,#line do
        line[j] = tonumber(line[j])
    end
    -- store key points for each image into a table
    if pts[line[1]] == nil then
        pts[line[1]] = { {select(2, unpack(line))} }
    else 
        table.insert(pts[line[1]], {select(2, unpack(line))})
    end
end
-- close key points file
keypoints:close()

-- 
for i = 1, images:size(1) do
	images[i] = image_kpts(images[i],i-1)
end

-- save modified images to file
torch.save("data/5_expression_data_modif.t7", images)