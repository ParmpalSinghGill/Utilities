# import the necessary packages
import numpy as np



def IOU(bboxes1, bboxes2):
	x11, y11, x12, y12 = bboxes1
	x21, y21, x22, y22 = bboxes2
	xA = np.maximum(x11, np.transpose(x21))
	yA = np.maximum(y11, np.transpose(y21))
	xB = np.minimum(x12, np.transpose(x22))
	yB = np.minimum(y12, np.transpose(y22))
	interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)
	boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
	boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
	iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)
	return iou

def NonMaXSuppression(boxes,scores,maxbox,iout):
	included=[]
	sindex = np.argsort(scores)[::-1]
	while len(sindex)>0:
		included.append(sindex[0])
		discardesd=[sindex[0]]
		for i in range(1,len(sindex)):
			if IOU(boxes[0],boxes[i])>iout:
				discardesd.append(i)
		sindex=np.delete(sindex,discardesd)
	return included

